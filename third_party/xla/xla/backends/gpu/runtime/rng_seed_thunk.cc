/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/rng_seed_thunk.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_address.h"
#include "tsl/platform/random.h"

namespace xla::gpu {

uint64_t RngSeedThunk::ResolveSeed(const Thunk::ExecuteParams& params) const {
  uint64_t seed = params.rng_seed;
  if (seed == 0) {
    // 0 is avoided because it is used as a sentinel value indicating that a
    // random seed should be generated. Generate a random non-zero seed as
    // fallback.
    do {
      seed = tsl::random::New64();
    } while (seed == 0);
  }
  return seed;
}

absl::Status RngSeedThunk::ExecuteOnStream(const Thunk::ExecuteParams& params) {
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);
  uint64_t seed = ResolveSeed(params);
  VLOG(3) << "RngSeedThunk executing with seed " << seed;
  return params.stream->Memcpy(&dest_addr, &seed, sizeof(uint64_t));
}

}  // namespace xla::gpu
