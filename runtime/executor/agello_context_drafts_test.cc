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

}  // namespace
}  // namespace litert::lm
