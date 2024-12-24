/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"
#include "Tensor.h"

using namespace TinyGPT;

TEST(TEST_TENSOR, constructor_default) {
  Tensor x;

  EXPECT_TRUE(x.empty());
  EXPECT_FALSE(x.isScalar());
  EXPECT_TRUE(x.dim() == 0);
}

TEST(TEST_TENSOR, constructor_shape) {
  Tensor x = Tensor::shape({2, 3});

  EXPECT_FALSE(x.empty());
  EXPECT_FALSE(x.isScalar());

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TENSOR, constructor_scalar) {
  Tensor x = Tensor::scalar(2);

  EXPECT_FALSE(x.empty());
  EXPECT_TRUE(x.isScalar());

  EXPECT_TRUE(x.dim() == 0);
  EXPECT_TRUE(x.size() == 1);
  EXPECT_THAT(x.toArray(), ElementsAre(2));
}

TEST(TEST_TENSOR, constructor_ones) {
  Tensor x = Tensor::ones({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST(TEST_TENSOR, constructor_zeros) {
  Tensor x = Tensor::zeros({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(TEST_TENSOR, constructor_tri) {
  Tensor x = Tensor::tri(3);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 0, 0, 1, 1, 0, 1, 1, 1));

  x = Tensor::tri(2, 3);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 0, 0, 1, 1, 0));

  x = Tensor::tri(3, 3, 1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 1, 0, 1, 1, 1, 1, 1, 1));

  x = Tensor::tri(3, 3, -1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(0, 0, 0, 1, 0, 0, 1, 1, 0));
}

TEST(TEST_TENSOR, constructor_1d) {
  Tensor x({1, 2, 3});

  EXPECT_TRUE(x.dim() == 1);
  EXPECT_TRUE(x.size() == 3);
  EXPECT_THAT(x.shape(), ElementsAre(3));
  EXPECT_THAT(x.strides(), ElementsAre(1));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 2, 3));
}

TEST(TEST_TENSOR, constructor_2d) {
  Tensor x({{1, 2}, {3, 4}, {5, 6}});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(3, 2));
  EXPECT_THAT(x.strides(), ElementsAre(2, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_TENSOR, constructor_3d) {
  Tensor x({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});

  EXPECT_TRUE(x.dim() == 3);
  EXPECT_TRUE(x.size() == 12);
  EXPECT_THAT(x.shape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(6, 3, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(4, 2, 3, 1, 0, 3, 4, 2, 3, 1, 0, 3));
}

TEST(TEST_TENSOR, basic_range) {
  auto range = Tensor::range(3, 6);
  EXPECT_THAT(range, ElementsAre(3, 4, 5));

  range = Tensor::range(3, 10, 2);
  EXPECT_THAT(range, ElementsAre(3, 5, 7, 9));

  auto t = Tensor::arange(3, 10, 2);
  EXPECT_THAT(t.shape(), ElementsAre(4));
  EXPECT_THAT(t.toArray(), ElementsAre(3, 5, 7, 9));
}

TEST(TEST_TENSOR, basic_indexing) {
  Tensor x({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  auto y = x.index({-1, 0});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(7, 8, 9, 1, 2, 3));

  y = x.index(std::vector<int32_t>{1});
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(4, 5, 6));
}

TEST(TEST_TENSOR, basic_transpose) {
  Tensor x({1, 2, 3});
  auto y = x.transpose();
  EXPECT_TRUE(y.shape() == x.shape());
  EXPECT_TRUE(y.toArray() == x.toArray());

  x = Tensor({{1, 2}, {3, 4}, {5, 6}});
  y = x.transpose();
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 3, 5, 2, 4, 6));
}

TEST(TEST_TENSOR, basic_split) {
  Tensor x({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
  auto y = x.split(2, 0);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 2, 3, 1, 0, 3));
  EXPECT_THAT(y[1].shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y[1].toArray(), ElementsAre(4, 2, 3, 1, 0, 3));

  y = x.split(2, 1);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 2, 3, 4, 2, 3));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y[1].toArray(), ElementsAre(1, 0, 3, 1, 0, 3));

  y = x.split(3, 2);
  EXPECT_TRUE(y.size() == 3);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 1, 4, 1));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[1].toArray(), ElementsAre(2, 0, 2, 0));
  EXPECT_THAT(y[2].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[2].toArray(), ElementsAre(3, 3, 3, 3));

  y = x.split(std::vector<int32_t>({1}), 2);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 1, 4, 1));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(y[1].toArray(), ElementsAre(2, 3, 0, 3, 2, 3, 0, 3));
}

TEST(TEST_TENSOR, basic_concatenate) {
  Tensor a(Array2d({{1, 2}, {3, 4}}));
  Tensor b(Array2d({{5, 6}}));
  Tensor bT = b.transpose();
  auto y = Tensor::concatenate({a, b}, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = Tensor::concatenate({a, bT}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 5, 3, 4, 6));

  y = Tensor::concatenate({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_TENSOR, basic_stack) {
  Tensor a({1, 2, 3});
  Tensor b({4, 5, 6});
  auto y = Tensor::stack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = Tensor::stack({a, b}, -1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = Tensor::stack({a, b}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(TEST_TENSOR, basic_vstack) {
  Tensor a({1, 2, 3});
  Tensor b({4, 5, 6});
  auto y = Tensor::vstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  a = Tensor(Array2d({{1}, {2}, {3}}));
  b = Tensor(Array2d({{4}, {5}, {6}}));
  y = Tensor::vstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_TENSOR, basic_hstack) {
  Tensor a({1, 2, 3});
  Tensor b({4, 5, 6});
  auto y = Tensor::hstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  a = Tensor(Array2d({{1}, {2}, {3}}));
  b = Tensor(Array2d({{4}, {5}, {6}}));
  y = Tensor::hstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(TEST_TENSOR, basic_dstack) {
  Tensor a({1, 2, 3});
  Tensor b({2, 3, 4});
  auto y = Tensor::dstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(1, 3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 2, 3, 3, 4));

  a = Tensor(Array2d({{1}, {2}, {3}}));
  b = Tensor(Array2d({{2}, {3}, {4}}));
  y = Tensor::dstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(3, 1, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 2, 3, 3, 4));
}

TEST(TEST_TENSOR, math_scalar) {
  Tensor x({{1, 2}, {3, 4}});

  // add
  Tensor y = 2 + x + 1.5;
  EXPECT_THAT(y.toArray(), ElementsAre(4.5, 5.5, 6.5, 7.5));
  y += 0.5;
  EXPECT_THAT(y.toArray(), ElementsAre(5, 6, 7, 8));

  // sub
  y = 2 - x - 1.5;
  EXPECT_THAT(y.toArray(), ElementsAre(-0.5, -1.5, -2.5, -3.5));
  y -= 0.5;
  EXPECT_THAT(y.toArray(), ElementsAre(-1, -2, -3, -4));

  // mul
  y = 2 * x * 1.5;
  EXPECT_THAT(y.toArray(), ElementsAre(3, 6, 9, 12));
  y *= 2;
  EXPECT_THAT(y.toArray(), ElementsAre(6, 12, 18, 24));

  // div
  y = 12 / x / 2;
  EXPECT_THAT(y.toArray(), ElementsAre(6, 3, 2, 1.5));
  y /= 0.5;
  EXPECT_THAT(y.toArray(), ElementsAre(12, 6, 4, 3));
}

TEST(TEST_TENSOR, math_same_shape) {
  Tensor x1({{1, 2}, {3, 4}});
  Tensor x2({{2, 3}, {4, 5}});

  auto y = x1 + x2;
  EXPECT_THAT(y.toArray(), ElementsAre(3, 5, 7, 9));
  y += x1;
  EXPECT_THAT(y.toArray(), ElementsAre(4, 7, 10, 13));

  y = x1 - x2;
  EXPECT_THAT(y.toArray(), ElementsAre(-1, -1, -1, -1));
  y -= x1;
  EXPECT_THAT(y.toArray(), ElementsAre(-2, -3, -4, -5));

  y = x1 * x2;
  EXPECT_THAT(y.toArray(), ElementsAre(2, 6, 12, 20));
  y *= x1;
  EXPECT_THAT(y.toArray(), ElementsAre(2, 12, 36, 80));

  y = x1 / x2;
  EXPECT_THAT(y.toArray(), ElementsAre(0.5, 2.f / 3, 0.75, 0.8));
  y /= x1;
  EXPECT_THAT(y.toArray(), ElementsAre(0.5, 1.f / 3, 0.25, 0.2));

  x1 = Tensor::scalar(1.f);
  x2 = Tensor::scalar(2.f);
  y = x1 - x2;
  EXPECT_THAT(y.toArray(), ElementsAre(-1));
}

TEST(TEST_TENSOR, math_min) {
  Tensor x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(Tensor::min(x) == 1);

  auto y = Tensor::min(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3));

  y = Tensor::min(x, 0, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3));

  y = Tensor::min(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));

  y = Tensor::min(x, 1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));

  y = Tensor::min(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));

  y = Tensor::min(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));
}

TEST(TEST_TENSOR, math_max) {
  Tensor x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(Tensor::max(x) == 6);

  auto y = Tensor::max(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(4, 5, 6));

  y = Tensor::max(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 6));
}

TEST(TEST_TENSOR, math_meam) {
  Tensor x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(Tensor::mean(x) == 3.5);

  auto y = Tensor::mean(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(2.5, 3.5, 4.5));

  y = Tensor::mean(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(2, 5));
}

TEST(TEST_TENSOR, math_sum) {
  Tensor x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(Tensor::sum(x) == 21);

  auto y = Tensor::sum(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(5, 7, 9));

  y = Tensor::sum(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(6, 15));

  x = Tensor({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
  EXPECT_TRUE(Tensor::sum(x) == 26);

  y = Tensor::sum(x, 2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(9, 4, 9, 4));

  y = Tensor::sum(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(5, 2, 6, 5, 2, 6));

  y = Tensor::sum(x, 0, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(8, 4, 6, 2, 0, 6));
}

TEST(TEST_TENSOR, math_var) {
  Tensor x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_FLOAT_NEAR(Tensor::var(x), 2.9166666);

  auto y = Tensor::var(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(2.25, 2.25, 2.25));

  y = Tensor::var(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_FLOAT_NEAR(y[0], 0.666667);
  EXPECT_FLOAT_NEAR(y[1], 0.666667);
}

TEST(TEST_TENSOR, math_argmin) {
  Tensor x({{4, 2, 3}, {1, 0, 3}});

  EXPECT_TRUE(Tensor::argmin(x) == 4);

  auto y = Tensor::argmin(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 1));

  y = Tensor::argmin(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 1));
}

TEST(TEST_TENSOR, math_argmax) {
  Tensor x({{1, 2, 4}, {1, 0, 3}});

  EXPECT_TRUE(Tensor::argmax(x) == 2);

  auto y = Tensor::argmax(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(2, 2));

  y = Tensor::argmax(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(2, 2));
}

TEST(TEST_TENSOR, math_sqrt) {
  Tensor x({{1, 2}, {3, 4}});
  auto y = Tensor::sqrt(x);
  EXPECT_THAT(y.toArray(), ElementsAre(std::sqrt(1),
                                    std::sqrt(2),
                                    std::sqrt(3),
                                    std::sqrt(4)));
}

TEST(TEST_TENSOR, math_tanh) {
  Tensor x({{1, 2}, {3, 4}});
  auto y = Tensor::tanh(x);
  EXPECT_NEAR(y[0], std::tanh(1), 1e-4);
  EXPECT_NEAR(y[1], std::tanh(2), 1e-4);
  EXPECT_NEAR(y[2], std::tanh(3), 1e-4);
  EXPECT_NEAR(y[3], std::tanh(4), 1e-4);
}

TEST(TEST_TENSOR, math_exp) {
  Tensor x({{1, 2}, {3, 4}});
  auto y = Tensor::exp(x);
  EXPECT_THAT(y.toArray(), ElementsAre(std::exp(1),
                                    std::exp(2),
                                    std::exp(3),
                                    std::exp(4)));
}

TEST(TEST_TENSOR, math_dot) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = Tensor::dot(Tensor(d1), Tensor(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(10, 13, 22, 29));

  Array1d d3 = {1, 2, 3};
  y = Tensor::dot(Tensor(d3), Tensor(d3));
  EXPECT_TRUE(y.isScalar());
  EXPECT_THAT(y.toArray(), ElementsAre(14));

  y = Tensor::dot(Tensor(d3), 0.2f);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(0.2f, 0.4f, 0.6f));

  y = Tensor::dot(0.2f, Tensor(d3));
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(0.2f, 0.4f, 0.6f));
}

TEST(TEST_TENSOR, math_matmul) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = Tensor::matmul(Tensor(d1), Tensor(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(10, 13, 22, 29));

  Array2d d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d d4 = {{2, 3}, {4, 5}, {6, 7}};
  y = Tensor::matmul(Tensor(d3), Tensor(d4));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(28, 34, 64, 79));

  Array2d d5 = {{1, 0}, {0, 1}};
  Array1d d6 = {1, 2};
  y = Tensor::matmul(Tensor(d5), Tensor(d6));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2));

  y = Tensor::matmul(Tensor(d6), Tensor(d5));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2));

  Array1d d7 = {2};
  y = Tensor::matmul(Tensor(d7), Tensor(d7));
  EXPECT_TRUE(y.isScalar());
  EXPECT_THAT(y.toArray(), ElementsAre(4));

  // broadcast
  auto a = Tensor::arange(0, 2 * 2 * 4).reshape({2, 2, 4});
  auto b = Tensor::arange(0, 2 * 2 * 4).reshape({1, 2, 4, 2});
  auto c = Tensor::arange(0, 1 * 2 * 4).reshape({1, 4, 2});
  auto d = Tensor::matmul(a, b);
  auto e = Tensor::matmul(a, c);

  EXPECT_THAT(d.shape(), ElementsAre(1, 2, 2, 2));
  EXPECT_THAT(d.toArray(), ElementsAre(28, 34, 76, 98, 428, 466, 604, 658));

  EXPECT_THAT(e.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(e.toArray(), ElementsAre(28, 34, 76, 98, 124, 162, 172, 226));
}

TEST(TEST_TENSOR, math_matmulTrans) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = Tensor::matmulTrans(Tensor(d1), Tensor(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(8, 14, 18, 32));

  Array2d d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d d4 = {{2, 4, 6}, {3, 5, 7}};
  y = Tensor::matmulTrans(Tensor(d3), Tensor(d4));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(28, 34, 64, 79));
}

TEST(TEST_TENSOR, math_broadcast) {
  Array2d d1 = {{1, 2}};
  Array2d d2 = {{2, 3}, {4, 5}};
  Array2d d3 = {{2}, {4}};
  Array1d d4 = {1, 2};
  Array1d d5 = {1};

  auto y = Tensor(d1) + Tensor(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 5, 5, 7));

  y = Tensor(d2) + Tensor(d3);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(4, 5, 8, 9));

  y = Tensor(d2) + Tensor(d4);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 5, 5, 7));

  y = Tensor(d2) + Tensor(d5);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 4, 5, 6));

  y = Tensor(d2) + Tensor::scalar(0.5);
  EXPECT_THAT(y.toArray(), ElementsAre(2.5, 3.5, 4.5, 5.5));
}
