// Copyright 2026 Patrice Déry (Agello, PowerBeef/agello-litert-lm)
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

#import "agello/agello_telemetry_metal.h"

#import <Foundation/Foundation.h>

#include <atomic>
#include <mutex>

#include "agello/agello_telemetry.h"

namespace agello::telemetry {

void AttachGpuTiming(id<MTLCommandBuffer> command_buffer) {
  const int level = GetLevel();
  if (level < kCounters || command_buffer == nil) return;
  Count(kCommandBuffers);
  // Now and then: what the GPU has allocated, and how the device is doing.
  static std::atomic<uint64_t> calls{0};
  if (calls.fetch_add(1, std::memory_order_relaxed) % 64 == 0) {
    NSProcessInfo* process = NSProcessInfo.processInfo;
    RecordDeviceState(static_cast<int64_t>(command_buffer.device.currentAllocatedSize),
                      static_cast<int>(process.thermalState), process.lowPowerModeEnabled);
  }
  const int run = CurrentRun();
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
    if (buffer.status == MTLCommandBufferStatusError) {
      ReportAnomaly(kCommandBufferError, kError, static_cast<int64_t>(buffer.error.code), run,
                    "Metal command buffer failed");
    }
    if (GetLevel() < kSpans) return;
    RecordGpuSeconds(run, buffer.GPUStartTime, buffer.GPUEndTime);
    // Submit to start: includes waiting behind earlier command buffers of the same run.
    RecordGpuSeconds(kGpuQueueDelay, buffer.kernelStartTime, buffer.GPUStartTime);
  }];
}

namespace {
std::mutex hold_mutex;
id<MTLSharedEvent> hold_event = nil;
uint64_t hold_value = 0;

void ReleaseHold() {
  std::lock_guard<std::mutex> lock(hold_mutex);
  if (hold_event != nil) hold_event.signaledValue = hold_value;
}
}  // namespace

void EncodeHoldIfTaken(id<MTLCommandBuffer> command_buffer) {
  if (!TakeGpuRunHold()) return;
  std::lock_guard<std::mutex> lock(hold_mutex);
  if (hold_event == nil) hold_event = [command_buffer.device newSharedEvent];
  ++hold_value;
  [command_buffer encodeWaitForEvent:hold_event value:hold_value];
  SetHoldReleaseHook(&ReleaseHold);
}

}  // namespace agello::telemetry
