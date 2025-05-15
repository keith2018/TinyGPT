/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "tokenizer/Base.h"

#pragma warning(disable : 4244)

using namespace testing;

namespace tinygpt {

void checkFloatVectorNear(const std::vector<float>& v1, const std::vector<float>& v2);

#define FLOAT_ABS_ERROR 1e-3
#define EXPECT_FLOAT_NEAR(v1, v2) EXPECT_NEAR(v1, v2, FLOAT_ABS_ERROR)

#define EXPECT_FLOAT_VEC_NEAR checkFloatVectorNear

std::vector<std::string> getStrings(const tokenizer::StringPieces& input);

}  // namespace tinygpt