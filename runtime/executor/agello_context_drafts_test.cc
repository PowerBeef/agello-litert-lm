// Copyright 2026 Patrice Déry
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/agello_context_drafts.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Syncs to the whole of `tokens`, the last one as the round's input token.
void SyncTo(ContextDrafts& drafts, const std::vector<int>& tokens) {
  drafts.Sync(absl::MakeConstSpan(tokens).first(tokens.size() - 1), tokens.back());
}

TEST(ContextDraftsTest, ProposesWhatFollowedTheLatestTrigram) {
  ContextDrafts drafts(3);
  // "1 2 3 4 5 6 | 9 1 2 3": the context's last trigram (1 2 3) was followed by 4 5 6.
  SyncTo(drafts, {1, 2, 3, 4, 5, 6, 9, 1, 2, 3});
  int ngram = 0;
  EXPECT_THAT(drafts.Propose(3, &ngram), ElementsAre(4, 5, 6));
  EXPECT_EQ(ngram, 3);
}

TEST(ContextDraftsTest, TheLatestOccurrenceWins) {
  ContextDrafts drafts(3);
  SyncTo(drafts, {1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 0, 1, 2, 3});
  EXPECT_THAT(drafts.Propose(3), ElementsAre(7, 8, 9));
}

TEST(ContextDraftsTest, NothingWithoutAnEarlierOccurrenceOrEnoughTokensAfterIt) {
  ContextDrafts drafts(3);
  SyncTo(drafts, {1, 2, 3, 4, 5, 6});
  EXPECT_THAT(drafts.Propose(3), IsEmpty());
  // The trigram 4 5 6 recurs; five tokens follow its earlier occurrence (into the recurrence).
  SyncTo(drafts, {4, 5, 6, 7, 8, 4, 5, 6});
  EXPECT_THAT(drafts.Propose(2), ElementsAre(7, 8));
  EXPECT_THAT(drafts.Propose(5), ElementsAre(7, 8, 4, 5, 6));
  EXPECT_THAT(drafts.Propose(6), IsEmpty());
}

TEST(ContextDraftsTest, BigramsOnlyWhenAllowed) {
  // The last bigram (2 3) occurred before; the last trigram (9 2 3) didn't.
  const std::vector<int> context = {2, 3, 4, 5, 6, 9, 2, 3};
  ContextDrafts trigrams(3);
  SyncTo(trigrams, context);
  EXPECT_THAT(trigrams.Propose(3), IsEmpty());
  ContextDrafts bigrams(2);
  SyncTo(bigrams, context);
  int ngram = 0;
  EXPECT_THAT(bigrams.Propose(3, &ngram), ElementsAre(4, 5, 6));
  EXPECT_EQ(ngram, 2);
}

TEST(ContextDraftsTest, FollowsAGrowingContextWithoutRebuilding) {
  ContextDrafts drafts(3);
  std::vector<int> context = {10, 11, 12, 13, 14, 15, 16};
  SyncTo(drafts, context);
  for (int token : {20, 10, 11, 12}) {
    context.push_back(token);
    SyncTo(drafts, context);
  }
  EXPECT_EQ(drafts.rebuilds(), 0);
  EXPECT_EQ(drafts.size(), static_cast<int>(context.size()));
  EXPECT_THAT(drafts.Propose(3), ElementsAre(13, 14, 15));
}

TEST(ContextDraftsTest, RebuildsAfterARewind) {
  ContextDrafts drafts(3);
  SyncTo(drafts, {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3});
  // A rewind to 5 tokens, then another continuation: what followed (1 2 3) before is gone.
  SyncTo(drafts, {1, 2, 3, 4, 5, 9, 1, 2, 3});
  EXPECT_EQ(drafts.rebuilds(), 1);
  EXPECT_THAT(drafts.Propose(3), ElementsAre(4, 5, 9));
  // The same length with another input token is a rebuild too.
  SyncTo(drafts, {1, 2, 3, 4, 5, 9, 1, 2, 7});
  EXPECT_EQ(drafts.rebuilds(), 2);
  EXPECT_THAT(drafts.Propose(3), IsEmpty());
}

// B5 phase 3: a reply copying the context realigns past an edit as soon as it writes a token
// the context has again, instead of three tokens later.
TEST(ContextDraftsTest, RealignsAfterAnEdit) {
  ContextDrafts drafts(3);
  // The source (1 … 12), then a reply that has copied its start.
  std::vector<int> context = {30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 40, 1, 2, 3};
  SyncTo(drafts, context);
  EXPECT_THAT(drafts.Propose(3), ElementsAre(4, 5, 6));
  // The reply keeps 4, then writes 50 where the source has 5: an edit.
  context.insert(context.end(), {4, 50});
  SyncTo(drafts, context);
  EXPECT_THAT(drafts.Propose(3), IsEmpty());
  // The next token the source has again (6) realigns the copy.
  context.push_back(6);
  SyncTo(drafts, context);
  int ngram = 0;
  EXPECT_THAT(drafts.Propose(3, &ngram), ElementsAre(7, 8, 9));
  EXPECT_EQ(ngram, 1);
}

TEST(ContextDraftsTest, RealignmentAllowsExactEndOfContext) {
  ContextDrafts drafts(3);
  // Start copying (1, 2, 3) -> proposes (4, 5, 6), then keeps 4, diverges at 50, realigns at 6.
  // Source ends with 6, 7, 8, 9, 40, 1, 2, 3, 4, 50, 6; asking for count reaching `size` must work.
  std::vector<int> context = {1, 2, 3, 4, 5, 6, 40, 1, 2, 3};
  SyncTo(drafts, context);
  EXPECT_THAT(drafts.Propose(3), ElementsAre(4, 5, 6));
  context.insert(context.end(), {4, 50});
  SyncTo(drafts, context);
  context.push_back(6);
  SyncTo(drafts, context);
  // Token 6 is at index 5 (`at = 5`), `size = 13`, so `at + 1 + 7 == 13 == size`.
  int ngram = 0;
  EXPECT_THAT(drafts.Propose(7, &ngram), ElementsAre(40, 1, 2, 3, 4, 50, 6));
  EXPECT_EQ(ngram, 1);
  EXPECT_THAT(drafts.Propose(8), IsEmpty());
}

TEST(ContextDraftsTest, NoRealignmentWithoutACopy) {
  ContextDrafts drafts(3);
  // 6 occurs earlier, but nothing was being copied: no proposal on one token.
  SyncTo(drafts, {1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 6});
  EXPECT_THAT(drafts.Propose(3), IsEmpty());
}

TEST(ContextDraftsTest, NeverProposesNegativeMultimodalSentinels) {
  ContextDrafts drafts(2);
  // Vision (-1) and audio (-2) sentinels follow (1, 2, 3); recurring (1, 2, 3)
  // must never propose them.
  SyncTo(drafts, {1, 2, 3, -1, -1, -2, 7, 8, 9, 1, 2, 3});
  EXPECT_THAT(drafts.Propose(3), IsEmpty());
  EXPECT_THAT(drafts.Propose(1), IsEmpty());

  // A candidate whose continuation contains a sentinel mid-window is also refused,
  // while a shorter window before the sentinel remains valid.
  ContextDrafts mid(3);
  SyncTo(mid, {1, 2, 3, 4, -1, 6, 9, 1, 2, 3});
  EXPECT_THAT(mid.Propose(3), IsEmpty());
  EXPECT_THAT(mid.Propose(2), IsEmpty());
  EXPECT_THAT(mid.Propose(1), ElementsAre(4));

  // A later sentinel continuation does not clobber an earlier valid continuation.
  ContextDrafts preserve(3);
  SyncTo(preserve, {1, 2, 3, 4, 5, 6, 9, 1, 2, 3, -2, -2, 8, 1, 2, 3});
  EXPECT_THAT(preserve.Propose(3), ElementsAre(4, 5, 6));
}

TEST(ContextDraftsTest, BreaksNgramIndexingAndRealignmentAtNegativeSentinels) {
  // N-grams containing -1/-2 are not indexed and do not match.
  ContextDrafts drafts(2);
  SyncTo(drafts, {1, -1, 2, 4, 5, 6, 9, 1, -1, 2});
  EXPECT_THAT(drafts.Propose(3), IsEmpty());

  // Realignment stops at negative sentinels and never emits them.
  ContextDrafts realign_boundary(3);
  std::vector<int> context = {30, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 40, 1, 2, 3};
  SyncTo(realign_boundary, context);
  EXPECT_THAT(realign_boundary.Propose(2), ElementsAre(4, 5));
  context.insert(context.end(), {4, 50});
  SyncTo(realign_boundary, context);
  // Token 6 lies past the -1 sentinel after diverged_at_; must not cross it.
  context.push_back(6);
  SyncTo(realign_boundary, context);
  EXPECT_THAT(realign_boundary.Propose(3), IsEmpty());

  // Realignment onto a valid token whose continuation includes a sentinel is refused.
  ContextDrafts realign_tail(3);
  std::vector<int> tail_ctx = {30, 1, 2, 3, 4, 5, 6, 7, -2, 9, 40, 1, 2, 3};
  SyncTo(realign_tail, tail_ctx);
  EXPECT_THAT(realign_tail.Propose(2), ElementsAre(4, 5));
  tail_ctx.insert(tail_ctx.end(), {4, 50});
  SyncTo(realign_tail, tail_ctx);
  tail_ctx.push_back(6);
  SyncTo(realign_tail, tail_ctx);
  EXPECT_THAT(realign_tail.Propose(3), IsEmpty());
  EXPECT_THAT(realign_tail.Propose(1), ElementsAre(7));
}

}  // namespace
}  // namespace litert::lm
