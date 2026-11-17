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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AGELLO_CONTEXT_DRAFTS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AGELLO_CONTEXT_DRAFTS_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Agello (B2, PERFORMANCE.md): drafts copied from the context. A reply that repeats its input (a
// summary's quotes, an edited document, an answer from a file) can be drafted by lookup: the
// tokens that followed the latest earlier occurrence of the context's last three tokens (or two).
// The index holds every bigram and trigram that has a continuation, the latest occurrence
// winning, and follows the context as it grows; a context that doesn't extend the last one (a
// rewind, another conversation) is indexed again.
class ContextDrafts {
 public:
  // `min_ngram` is the shortest match a proposal may rest on: 2 or 3.
  explicit ContextDrafts(int min_ngram) : min_ngram_(min_ngram) {}

  // Brings the index to `prefix` followed by `last` (the round's input token).
  void Sync(absl::Span<const int> prefix, int last);

  // The `count` tokens that followed the latest earlier occurrence of the context's last three
  // tokens, or of its last two when `min_ngram` allows; `ngram` gets the n used. Empty when
  // there's no occurrence or fewer than `count` tokens follow it.
  std::vector<int> Propose(int count, int* ngram = nullptr) const;

  int size() const { return static_cast<int>(tokens_.size()); }
  // How many times the index was built from scratch (for tests and telemetry).
  int rebuilds() const { return rebuilds_; }

 private:
  // Indexes the n-grams continued by the tokens at `from` and after.
  void Index(int from);

  int min_ngram_;
  int rebuilds_ = 0;
  std::vector<int> tokens_;
  // An n-gram's key → the index of the token that followed its latest occurrence.
  absl::flat_hash_map<uint64_t, int> trigrams_;
  absl::flat_hash_map<uint64_t, int> bigrams_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AGELLO_CONTEXT_DRAFTS_H_
