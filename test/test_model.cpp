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
  EXPECT_FLOAT_NEAR(y[0], 0.84119);
  EXPECT_FLOAT_NEAR(y[1], 1.9546);
  EXPECT_FLOAT_NEAR(y[2], -0.0454);
  EXPECT_FLOAT_NEAR(y[3], 0.34571);
}

TEST(TEST_MODEL, basic_softmax) {
  Tensor x({{2, 3}, {2, 4}});
  auto y = Model::softmax(x);
  auto sum = Tensor::sum(y, -1);

  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_FLOAT_NEAR(y[0], 0.26894142);
  EXPECT_FLOAT_NEAR(y[1], 0.73105858);
  EXPECT_FLOAT_NEAR(y[2], 0.11920292);
  EXPECT_FLOAT_NEAR(y[3], 0.88079708);

  EXPECT_FLOAT_NEAR(sum[0], 1);
  EXPECT_FLOAT_NEAR(sum[1], 1);
}

TEST(TEST_MODEL, basic_layerNorm) {
  Tensor x({{2, 2, 3}, {-5, 0, 1}});
  auto g = Tensor::ones({x.shape().back()});
  auto b = Tensor::zeros({x.shape().back()});
  auto y = Model::layerNorm(x, g, b);

  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_FLOAT_NEAR(y[0], -0.70709087);
  EXPECT_FLOAT_NEAR(y[1], -0.70709087);
  EXPECT_FLOAT_NEAR(y[2], 1.41418174);
  EXPECT_FLOAT_NEAR(y[3], -1.39700038);
  EXPECT_FLOAT_NEAR(y[4], 0.50800014);
  EXPECT_FLOAT_NEAR(y[5], 0.88900024);
}

TEST(TEST_MODEL, basic_linear) {
  Array2d x = {{1, 2, 3}, {4, 5, 6}};
  Array2d w = {{1, 2}, {3, 4}, {5, 6}};
  Array1d b = {1.5, 2.5};

  auto y = Model::linear(Tensor(x), Tensor(w), Tensor(b));
  EXPECT_THAT(y.toArray(), ElementsAre(23.5, 30.5, 50.5, 66.5));
}
