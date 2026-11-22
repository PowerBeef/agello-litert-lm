/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tflite/kernels/cpu_backend_context.h"

#include <atomic>
#include <memory>
#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#include <sched.h>
#endif

#ifdef TFLITE_KERNEL_USE_XNNPACK
#include "pthreadpool.h"  // from @pthreadpool
#endif

#ifdef TFLITE_HAVE_CPUINFO
#include "include/cpuinfo.h"
#endif

#include "public/gemmlowp.h"
#include "ruy/context.h"  // from @ruy
#include "ruy/path.h"  // from @ruy
#include "tflite/core/c/common.h"
#include "tflite/core/macros.h"
#include "tflite/external_cpu_backend_context.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/op_macros.h"

namespace {
const int kDefaultNumThreadpoolThreads = 1;

}  // namespace

namespace tflite {

// Use weak symbols if possible to dispatch to deprecated paths.
#if TFLITE_HAS_ATTRIBUTE_WEAK && !defined(__APPLE__)
extern TFLITE_ATTRIBUTE_WEAK bool UseGemmlowpOnX86();
#endif  // defined(TFLITE_HAS_ATTRIBUTE_WEAK) && !(__APPLE__)

#if defined(TFLITE_HAVE_CPUINFO)
CpuBackendContext::CpuInfo::~CpuInfo() {
  if (init_status_ == InitStatus::kInitialized) {
    cpuinfo_deinitialize();
  }
}

bool CpuBackendContext::CpuInfo::EnsureInitialized() {
  if (init_status_ == InitStatus::kNotYetAttempted) {
    init_status_ = Initialize();
  }
  return init_status_ == InitStatus::kInitialized;
}

CpuBackendContext::CpuInfo::InitStatus
CpuBackendContext::CpuInfo::Initialize() {
  TFLITE_DCHECK_EQ(init_status_, InitStatus::kNotYetAttempted);
  if (!cpuinfo_initialize()) {
    return InitStatus::kFailed;
  }
  return InitStatus::kInitialized;
}

bool CpuBackendContext::CpuInfo::Avx2Fma() {
  return EnsureInitialized() && cpuinfo_has_x86_avx2() &&
         cpuinfo_has_x86_fma3();
}

bool CpuBackendContext::CpuInfo::Avx() {
  return EnsureInitialized() && cpuinfo_has_x86_avx();
}

bool CpuBackendContext::CpuInfo::Avx512() {
  return EnsureInitialized() && cpuinfo_has_x86_avx512f() &&
         cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512cd() &&
         cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512vl();
}
#else

CpuBackendContext::CpuInfo::~CpuInfo() {}

bool CpuBackendContext::CpuInfo::EnsureInitialized() {
  if (init_status_ == InitStatus::kNotYetAttempted) {
    init_status_ = InitStatus::kInitialized;
  }
  TFLITE_DCHECK_EQ(init_status_, InitStatus::kInitialized);
  return true;
}

bool CpuBackendContext::CpuInfo::Avx2Fma() { return false; }

bool CpuBackendContext::CpuInfo::Avx() { return false; }

bool CpuBackendContext::CpuInfo::Avx512() { return false; }
#endif  // TFLITE_HAVE_CPUINFO

CpuBackendContext* CpuBackendContext::GetFromContext(TfLiteContext* context) {
  auto* external_context = static_cast<ExternalCpuBackendContext*>(
      context->GetExternalContext(context, kTfLiteCpuBackendContext));

  if (external_context == nullptr) {
    TF_LITE_FATAL(
        "ExternalCpuBackendContext isn't properly initialized during TFLite "
        "interpreter initialization.");
  }

  auto* cpu_backend_context = static_cast<CpuBackendContext*>(
      external_context->internal_backend_context());
  if (cpu_backend_context == nullptr) {
    // We do the lazy initialization here for the TfLiteInternalBackendContext
    // that's wrapped inside ExternalCpuBackendContext.
    cpu_backend_context = new CpuBackendContext();
    cpu_backend_context->SetMaxNumThreads(context->recommended_num_threads);
    external_context->set_internal_backend_context(
        std::unique_ptr<TfLiteInternalBackendContext>(cpu_backend_context));
  }

  return cpu_backend_context;
}

CpuBackendContext::CpuBackendContext()
    : TfLiteInternalBackendContext(),
      ruy_context_(new ruy::Context),
      gemmlowp_context_(new gemmlowp::GemmContext) {
  SetMaxNumThreads(kDefaultNumThreadpoolThreads);
// TODO(b/148289189) Remove when clients have transitioned to runtime flag.
#ifdef TFLITE_WITH_RUY_GEMV
  SetUseCaching(true);
#else
  SetUseCaching(false);
#endif
}

CpuBackendContext::~CpuBackendContext() {}

void CpuBackendContext::SetMaxNumThreads(int max_num_threads) {
  const int target_num_threads =
      max_num_threads > -1 ? max_num_threads : kDefaultNumThreadpoolThreads;
  max_num_threads_ = target_num_threads;
  ruy_context_->set_max_num_threads(target_num_threads);
  gemmlowp_context_->set_max_num_threads(target_num_threads);
}

void CpuBackendContext::SetUseCaching(bool flag) { use_caching_ = flag; }

#ifdef TFLITE_KERNEL_USE_XNNPACK
pthreadpool_t CpuBackendContext::get_xnnpack_threadpool() {
  if (!xnnpack_threadpool_ && max_num_threads_ > 1) {
    xnnpack_threadpool_.reset(
        pthreadpool_create(static_cast<size_t>(max_num_threads_)));
#if defined(__APPLE__)
    // Agello (P4.2): promote XNNPACK worker pthreads to USER_INTERACTIVE so
    // CPU-delegated ops stay on P-cores.
    if (pthreadpool_t pool = xnnpack_threadpool_.get(); pool != nullptr) {
      const size_t n = pthreadpool_get_threads_count(pool);
      if (n > 1) {
        struct QoSContext {
          pthread_t caller;
          std::atomic<size_t> arrived{0};
          size_t target;
        } qos_ctx{pthread_self(), 0, n};
        pthreadpool_parallelize_1d(
            pool,
            [](void* raw_ctx, size_t /*i*/) {
              auto* c = static_cast<QoSContext*>(raw_ctx);
              if (!pthread_equal(pthread_self(), c->caller)) {
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
              }
              c->arrived.fetch_add(1, std::memory_order_acq_rel);
              for (int spins = 0;
                   spins < 200 &&
                   c->arrived.load(std::memory_order_acquire) < c->target;
                   ++spins) {
                sched_yield();
              }
            },
            &qos_ctx, n, /*flags=*/0);
      }
    }
#endif
  }
  return xnnpack_threadpool_.get();
}
#endif

bool CpuBackendContext::PreferGemmlowpOnX86() {
  bool use_gemmlowp_on_x86 = false;
#if defined(TFLITE_X86_PLATFORM) && TFLITE_HAS_ATTRIBUTE_WEAK && \
    !defined(__APPLE__)
  if (::tflite::UseGemmlowpOnX86 != nullptr) {
    use_gemmlowp_on_x86 = ::tflite::UseGemmlowpOnX86();
  }
#endif  // TFLITE_X86_PLATFORM && TFLITE_HAS_ATTRIBUTE_WEAK && !(__APPLE__)
  return use_gemmlowp_on_x86 || !RuyHasAvxOrAbove();
}

bool CpuBackendContext::RuyHasAvxOrAbove() {
  // TODO(b/183178387): Use a proper query to detect AVX/optimized paths.
#if RUY_PLATFORM_X86_ENHANCEMENTS
  return cpuinfo_.Avx() || cpuinfo_.Avx2Fma() || cpuinfo_.Avx512();
#else
  return false;
#endif
}

}  // namespace tflite
