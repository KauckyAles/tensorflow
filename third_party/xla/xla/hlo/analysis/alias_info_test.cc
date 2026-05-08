
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

#include "xla/hlo/analysis/alias_info.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

class AliasInfoTest : public HloHardwareIndependentTestBase {};

TEST_F(AliasInfoTest, AsyncStartDefaultAliasing) {
  const char* const kHlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0),
    to_apply=async_computation,
    output_to_operand_aliasing={{1}: (0, {})}
  ROOT done = f32[2,3] call-done(start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* start = FindInstruction(module.get(), "start");
  const HloInstruction* done = FindInstruction(module.get(), "done");
  AliasInfo alias_info;
  auto pairs_start = alias_info.GetInPlaceInputOutputPairs(start);
  EXPECT_TRUE(pairs_start.empty());
  auto pairs_done = alias_info.GetInPlaceInputOutputPairs(done);
  EXPECT_THAT(pairs_done, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                              HloOperandIndex{0, {0, 0}}, {}}));
}

// Tests that the alias info is computed correctly when the output-to-operand
// aliasing is late-bound.
TEST_F(AliasInfoTest, AsyncStart_LateBinding) {
  const char* const kHlo = R"(
HloModule test
async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}
ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((), (), s32[]) call-start(),
    to_apply=async_computation,
    output_to_operand_aliasing={{1}: (0, {})}
  update = ((f32[2,3]), f32[2,3]) async-update(start, p0)
  ROOT done = f32[2,3] call-done(update)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* start = FindInstruction(module.get(), "start");
  AliasInfo alias_info;
  auto pairs_start = alias_info.GetInPlaceInputOutputPairs(start);
  EXPECT_TRUE(pairs_start.empty());
  const HloInstruction* update = FindInstruction(module.get(), "update");
  auto pairs_update = alias_info.GetInPlaceInputOutputPairs(update);
  EXPECT_TRUE(pairs_update.empty());
  const HloInstruction* done = FindInstruction(module.get(), "done");
  auto pairs_done = alias_info.GetInPlaceInputOutputPairs(done);
  EXPECT_THAT(pairs_done, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                              HloOperandIndex{0, {0, 0}}, {}}));
}

}  // namespace
}  // namespace xla
