// Copyright 2026 Patrice Déry (Agello, PowerBeef/LiteRT)
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

// Agello (T11): the weight cache's content identifier ignores the sizes of
// tensors that aren't constants (the KV caches), and still follows the
// constant inputs.
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "tflite/c/common.h"

namespace ml_drift {
namespace {

// One node reading a KV cache (an arena tensor) and a weight (a constant).
struct FakeSubgraph {
  std::vector<TfLiteTensor> tensors = std::vector<TfLiteTensor>(3);
  TfLiteNode node{};
  TfLiteRegistration registration{};
  TfLiteContext context{};
  TfLiteDelegateParams params{};

  FakeSubgraph(size_t kv_bytes, size_t weight_bytes) {
    tensors[0].allocation_type = kTfLiteArenaRw;
    tensors[0].bytes = kv_bytes;
    tensors[0].type = kTfLiteInt8;
    tensors[1].allocation_type = kTfLiteMmapRo;
    tensors[1].bytes = weight_bytes;
    tensors[1].type = kTfLiteInt8;
    tensors[2].allocation_type = kTfLiteArenaRw;
    tensors[2].bytes = kv_bytes;
    tensors[2].type = kTfLiteFloat16;
    node.inputs = Ints({0, 1});
    node.outputs = Ints({2});
    registration.builtin_code = 9;  // FULLY_CONNECTED
    context.tensors = tensors.data();
    context.tensors_size = static_cast<int>(tensors.size());
    context.impl_ = this;
    context.GetNodeAndRegistration = [](TfLiteContext* context, int,
                                        TfLiteNode** node,
                                        TfLiteRegistration** registration) {
      auto* self = static_cast<FakeSubgraph*>(context->impl_);
      *node = &self->node;
      *registration = &self->registration;
      return kTfLiteOk;
    };
    params.nodes_to_replace = Ints({0});
    params.input_tensors = Ints({0});
    params.output_tensors = Ints({2});
  }
  ~FakeSubgraph() {
    TfLiteIntArrayFree(node.inputs);
    TfLiteIntArrayFree(node.outputs);
    TfLiteIntArrayFree(params.nodes_to_replace);
    TfLiteIntArrayFree(params.input_tensors);
    TfLiteIntArrayFree(params.output_tensors);
  }

  static TfLiteIntArray* Ints(std::vector<int> values) {
    TfLiteIntArray* array = TfLiteIntArrayCreate(static_cast<int>(values.size()));
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
      array->data[i] = values[i];
    }
    return array;
  }

  uint64_t Content(
      const std::unordered_map<size_t, size_t>* external_buffer_id_map =
          nullptr) {
    return SerializationWeightCache::GenerateContentModelIdentifier(
        "model", &context, &params, "prefix", kDefault,
        /*prefer_texture_weights=*/false,
        /*allow_src_quantized_fc_conv_ops=*/true,
        /*prepare_weights_in_batches=*/true,
        /*serialize_external_tensors=*/true, /*ordered_by_size=*/true,
        external_buffer_id_map);
  }
  uint64_t Unique() {
    return SerializationWeightCache::GenerateUniqueModelIdentifier(
        "model", &context, &params, "prefix", kDefault,
        /*prefer_texture_weights=*/false,
        /*allow_src_quantized_fc_conv_ops=*/true,
        /*prepare_weights_in_batches=*/true,
        /*serialize_external_tensors=*/true, /*ordered_by_size=*/true);
  }
};

TEST(ContentModelIdentifierTest, IgnoresTheKvCacheSize) {
  FakeSubgraph at_8k(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph at_32k(/*kv_bytes=*/32768, /*weight_bytes=*/4096);
  EXPECT_EQ(at_8k.Content(), at_32k.Content());
  // TFLite's fingerprint, the upstream key, tells them apart.
  EXPECT_NE(at_8k.Unique(), at_32k.Unique());
}

TEST(ContentModelIdentifierTest, FollowsTheConstantInputs) {
  FakeSubgraph a(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph b(/*kv_bytes=*/8192, /*weight_bytes=*/2048);
  EXPECT_NE(a.Content(), b.Content());
  FakeSubgraph c(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  c.registration.builtin_code = 3;  // CONV_2D
  EXPECT_NE(a.Content(), c.Content());
}

TEST(ContentModelIdentifierTest, FollowsDimsQuantExternalIdsAndNonKvIoShapes) {
  // Different weight dims with the same byte count must produce different IDs.
  FakeSubgraph shape_a(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph shape_b(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  shape_a.tensors[1].dims = FakeSubgraph::Ints({64, 64});
  shape_b.tensors[1].dims = FakeSubgraph::Ints({32, 128});
  EXPECT_NE(shape_a.Content(), shape_b.Content());
  TfLiteIntArrayFree(shape_a.tensors[1].dims);
  TfLiteIntArrayFree(shape_b.tensors[1].dims);

  // Different quantization parameters must produce different IDs.
  FakeSubgraph quant_a(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph quant_b(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  TfLiteFloatArray* scale_a = TfLiteFloatArrayCreate(1);
  scale_a->data[0] = 0.25f;
  TfLiteIntArray* zp_a = FakeSubgraph::Ints({0});
  TfLiteAffineQuantization aq_a{scale_a, zp_a, /*quantized_dimension=*/0};
  quant_a.tensors[1].quantization.type = kTfLiteAffineQuantization;
  quant_a.tensors[1].quantization.params = &aq_a;

  TfLiteFloatArray* scale_b = TfLiteFloatArrayCreate(1);
  scale_b->data[0] = 0.5f;
  TfLiteIntArray* zp_b = FakeSubgraph::Ints({0});
  TfLiteAffineQuantization aq_b{scale_b, zp_b, /*quantized_dimension=*/0};
  quant_b.tensors[1].quantization.type = kTfLiteAffineQuantization;
  quant_b.tensors[1].quantization.params = &aq_b;
  EXPECT_NE(quant_a.Content(), quant_b.Content());
  TfLiteFloatArrayFree(scale_a);
  TfLiteIntArrayFree(zp_a);
  TfLiteFloatArrayFree(scale_b);
  TfLiteIntArrayFree(zp_b);

  // Different external weight buffer IDs must produce different IDs.
  FakeSubgraph ext_a(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  std::unordered_map<size_t, size_t> ext_map_1{{1, 100}};
  std::unordered_map<size_t, size_t> ext_map_2{{1, 200}};
  EXPECT_NE(ext_a.Content(&ext_map_1), ext_a.Content(&ext_map_2));

  // Different non-KV subgraph I/O shapes must produce different IDs, while
  // KV cache I/O shapes remain ignored for cross-context-length sharing.
  FakeSubgraph prefill_128(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph decode_1(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  prefill_128.tensors[0].name = "embeddings";
  prefill_128.tensors[0].dims = FakeSubgraph::Ints({1, 128, 2560});
  decode_1.tensors[0].name = "embeddings";
  decode_1.tensors[0].dims = FakeSubgraph::Ints({1, 1, 2560});
  EXPECT_NE(prefill_128.Content(), decode_1.Content());
  TfLiteIntArrayFree(prefill_128.tensors[0].dims);
  TfLiteIntArrayFree(decode_1.tensors[0].dims);

  FakeSubgraph kv_8k(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph kv_64k(/*kv_bytes=*/65536, /*weight_bytes=*/4096);
  kv_8k.tensors[0].name = "kv_cache_k_0";
  kv_8k.tensors[0].dims = FakeSubgraph::Ints({1, 2, 8192, 512});
  kv_64k.tensors[0].name = "kv_cache_k_0";
  kv_64k.tensors[0].dims = FakeSubgraph::Ints({1, 2, 65536, 512});
  EXPECT_EQ(kv_8k.Content(), kv_64k.Content());
  TfLiteIntArrayFree(kv_8k.tensors[0].dims);
  TfLiteIntArrayFree(kv_64k.tensors[0].dims);
}

}  // namespace
}  // namespace ml_drift
