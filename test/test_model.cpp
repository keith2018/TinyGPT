/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"
#include "Model.h"

using namespace TinyGPT;

TEST(TEST_MODEL, basic_gelu) {
  Tensor x({{1, 2}, {-2, 0.5}});
  auto y = Model::gelu(x);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));

  auto arr = y.toList();
  EXPECT_FLOAT_NEAR(arr[0], 0.84119);
  EXPECT_FLOAT_NEAR(arr[1], 1.9546);
  EXPECT_FLOAT_NEAR(arr[2], -0.0454);
  EXPECT_FLOAT_NEAR(arr[3], 0.34571);
}

TEST(TEST_MODEL, basic_softmax) {
  Tensor x({{2, 3}, {2, 4}});
  auto y = Model::softmax(x);
  auto sum = Tensor::sum(y, -1);

  EXPECT_THAT(y.shape(), ElementsAre(2, 2));

  auto arr = y.toList();
  EXPECT_FLOAT_NEAR(arr[0], 0.26894142);
  EXPECT_FLOAT_NEAR(arr[1], 0.73105858);
  EXPECT_FLOAT_NEAR(arr[2], 0.11920292);
  EXPECT_FLOAT_NEAR(arr[3], 0.88079708);

  arr = sum.toList();
  EXPECT_FLOAT_NEAR(arr[0], 1);
  EXPECT_FLOAT_NEAR(arr[1], 1);
}

TEST(TEST_MODEL, basic_layerNorm) {
  Tensor x({{2, 2, 3}, {-5, 0, 1}});
  auto g = Tensor::ones({x.shape().back()});
  auto b = Tensor::zeros({x.shape().back()});
  auto y = Model::layerNorm(x, g, b);

  EXPECT_THAT(y.shape(), ElementsAre(2, 3));

  auto arr = y.toList();
  EXPECT_FLOAT_NEAR(arr[0], -0.70709087);
  EXPECT_FLOAT_NEAR(arr[1], -0.70709087);
  EXPECT_FLOAT_NEAR(arr[2], 1.41418174);
  EXPECT_FLOAT_NEAR(arr[3], -1.39700038);
  EXPECT_FLOAT_NEAR(arr[4], 0.50800014);
  EXPECT_FLOAT_NEAR(arr[5], 0.88900024);
}

TEST(TEST_MODEL, basic_linear) {
  TinyTorch::Array2d x = {{1, 2, 3}, {4, 5, 6}};
  TinyTorch::Array2d w = {{1, 2}, {3, 4}, {5, 6}};
  TinyTorch::Array1d b = {1.5, 2.5};

  auto y = Model::linear(Tensor(x), Tensor(w), Tensor(b));
  EXPECT_THAT(y.toList(), ElementsAre(23.5, 30.5, 50.5, 66.5));
}