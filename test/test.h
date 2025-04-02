/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"

#pragma warning(disable : 4244)

using namespace testing;

void checkFloatVectorNear(const std::vector<float>& v1,
    const std::vector<float>& v2);

#define FLOAT_ABS_ERROR 1e-3
#define EXPECT_FLOAT_NEAR(v1, v2) EXPECT_NEAR(v1, v2, FLOAT_ABS_ERROR)

#define EXPECT_FLOAT_VEC_NEAR checkFloatVectorNear
