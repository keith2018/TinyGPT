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

#define FLOAT_ABS_ERROR 1e-5
#define EXPECT_FLOAT_NEAR(v1, v2) EXPECT_NEAR(v1, v2, FLOAT_ABS_ERROR)
