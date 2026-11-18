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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
namespace {

// 21 bits a token: three fit in a key, for vocabularies up to 2M.
uint64_t Bits(int token) { return static_cast<uint64_t>(static_cast<uint32_t>(token)) & 0x1FFFFF; }

uint64_t Key(int a, int b) { return (Bits(a) << 21) | Bits(b); }

uint64_t Key(int a, int b, int c) { return (Bits(a) << 42) | (Bits(b) << 21) | Bits(c); }

}  // namespace

void ContextDrafts::Sync(absl::Span<const int> prefix, int last) {
  const size_t target = prefix.size() + 1;
  const size_t common = std::min(tokens_.size(), target);
  bool extends = tokens_.size() <= target &&
                 std::equal(tokens_.begin(),
                            tokens_.begin() + std::min(common, prefix.size()),
                            prefix.begin());
  if (extends && common == target) extends = tokens_.back() == last;
  if (!extends) {
    tokens_.clear();
    trigrams_.clear();
    bigrams_.clear();
    ++rebuilds_;
    proposed_from_ = proposed_at_size_ = diverged_at_ = -1;
  }
  const int from = static_cast<int>(tokens_.size());
  if (tokens_.size() < prefix.size()) {
    tokens_.insert(tokens_.end(), prefix.begin() + tokens_.size(), prefix.end());
  }
  if (tokens_.size() < target) tokens_.push_back(last);
  Index(from);
  // What the reply kept of the last proposal: where it stopped matching, the copy diverged.
  if (proposed_from_ >= 0 && proposed_at_size_ >= 0 &&
      static_cast<int>(tokens_.size()) > proposed_at_size_) {
    int kept = 0;
    while (proposed_at_size_ + kept < static_cast<int>(tokens_.size()) &&
           proposed_from_ + kept < proposed_at_size_ &&
           tokens_[proposed_at_size_ + kept] == tokens_[proposed_from_ + kept]) {
      ++kept;
    }
    diverged_at_ = kept > 0 ? proposed_from_ + kept : -1;
    proposed_from_ = proposed_at_size_ = -1;
  }
}

void ContextDrafts::Index(int from) {
  const int size = static_cast<int>(tokens_.size());
  for (int next = std::max(from, 2); next < size; ++next) {
    bigrams_[Key(tokens_[next - 2], tokens_[next - 1])] = next;
    if (next >= 3) trigrams_[Key(tokens_[next - 3], tokens_[next - 2], tokens_[next - 1])] = next;
  }
}

std::vector<int> ContextDrafts::Propose(int count, int* ngram) const {
  const int size = static_cast<int>(tokens_.size());
  auto from = [&](const absl::flat_hash_map<uint64_t, int>& map, uint64_t key) {
    std::vector<int> drafts;
    const auto found = map.find(key);
    if (found != map.end() && found->second + count <= size) {
      drafts.assign(tokens_.begin() + found->second, tokens_.begin() + found->second + count);
      proposed_from_ = found->second;
      proposed_at_size_ = size;
    }
    return drafts;
  };
  if (count <= 0) return {};
  if (size >= 3) {
    auto drafts = from(trigrams_, Key(tokens_[size - 3], tokens_[size - 2], tokens_[size - 1]));
    if (!drafts.empty()) {
      if (ngram != nullptr) *ngram = 3;
      return drafts;
    }
  }
  if (min_ngram_ <= 2 && size >= 2) {
    auto drafts = from(bigrams_, Key(tokens_[size - 2], tokens_[size - 1]));
    if (!drafts.empty()) {
      if (ngram != nullptr) *ngram = 2;
      return drafts;
    }
  }
  // Realign past an edit: the context's last token a few tokens after where the copy diverged.
  constexpr int kRealignWindow = 16;
  if (realign_ && diverged_at_ >= 0 && size >= 1) {
    const int last = tokens_[size - 1];
    const int end = std::min(diverged_at_ + kRealignWindow, size - 1);
    for (int at = diverged_at_; at < end; ++at) {
      if (tokens_[at] == last && at + 1 + count <= size - 1) {
        if (ngram != nullptr) *ngram = 1;
        proposed_from_ = at + 1;
        proposed_at_size_ = size;
        return std::vector<int>(tokens_.begin() + at + 1, tokens_.begin() + at + 1 + count);
      }
    }
  }
  return {};
}

}  // namespace litert::lm
