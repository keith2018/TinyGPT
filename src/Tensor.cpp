/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor.h"

#include <cassert>
#include <cstring>
#include <random>

#include "Blas.h"
#include "Logger.h"

namespace TinyGPT {

std::optional<unsigned int> RandomGenerator::seed_;
std::default_random_engine RandomGenerator::randomEngine_;

#define TENSOR_CHECK_EMPTY(t, ret)                                             \
  do {                                                                         \
    if ((t).empty()) {                                                         \
      Tensor::error(__FUNCTION__, TensorError_EmptyTensor);                    \
      return ret;                                                              \
    }                                                                          \
  } while (0)

#define TENSOR_CHECK_EMPTY_PAIR(t1, t2, ret)                                   \
  do {                                                                         \
    if ((t1).empty() || (t2).empty()) {                                        \
      error(__FUNCTION__, TensorError_EmptyTensor);                            \
      return ret;                                                              \
    }                                                                          \
  } while (0)

#define TENSOR_MATH_FAST_LOOP_SELF(op, other)                                  \
  do {                                                                         \
    for (int32_t idx = 0; idx < elemCount_; idx++) {                           \
      data_[idx] op other;                                                     \
    }                                                                          \
  } while (0)

#define TENSOR_MATH_FAST_LOOP_PAIR(init, op, other)                            \
  Tensor ret = init;                                                           \
  for (int32_t idx = 0; idx < ret.elemCount_; idx++) {                         \
    ret[idx] op other;                                                         \
  }                                                                            \
  return ret

#define TENSOR_CHECK_SHAPE_EQUAL_RET(ret)                                      \
  do {                                                                         \
    if (!shape_.empty() && !other.shape_.empty() && shape_ != other.shape_) {  \
      error(__FUNCTION__, TensorError_ShapeNotAligned);                        \
      return ret;                                                              \
    }                                                                          \
  } while (0)

#define TENSOR_CHECK_SHAPE_EQUAL                                               \
  do {                                                                         \
    if (!shape_.empty() && !other.shape_.empty() && shape_ != other.shape_) {  \
      error(__FUNCTION__, TensorError_ShapeNotAligned);                        \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define TENSOR_MATH_BROADCAST_PAIR(op)                                         \
  if (other.isScalar()) {                                                      \
    return *this op other[0];                                                  \
  }                                                                            \
  Shape retShape;                                                              \
  auto comp = checkCompatible(shape(), other.shape(), retShape);               \
  if (comp == ShapeCompatible_Error) {                                         \
    error(__FUNCTION__, TensorError_ShapeNotAligned);                          \
    return {};                                                                 \
  }                                                                            \
  if (comp == ShapeCompatible_SameShape) {                                     \
    TENSOR_MATH_FAST_LOOP_PAIR(*this, op## =, other[idx]);                     \
  }                                                                            \
  Tensor ret = shape(retShape);                                                \
  TensorIter it0(shape());                                                     \
  TensorIter it1(other.shape());                                               \
                                                                               \
  it0.broadcast(retShape);                                                     \
  it1.broadcast(retShape);                                                     \
                                                                               \
  for (int32_t idx = 0; idx < ret.elemCount_; idx++) {                         \
    ret[idx] = (*this)[it0.next()] op other[it1.next()];                       \
  }                                                                            \
  return ret

#define TENSOR_MATH_BROADCAST_PAIR_FUNC(op)                                    \
  Shape retShape;                                                              \
  auto comp = checkCompatible(shape(), other.shape(), retShape);               \
  if (comp == ShapeCompatible_Error) {                                         \
    error(__FUNCTION__, TensorError_ShapeNotAligned);                          \
    return {};                                                                 \
  }                                                                            \
  if (comp == ShapeCompatible_SameShape) {                                     \
    TENSOR_MATH_FAST_LOOP_PAIR(*this, =, op(ret[idx], other[idx]));            \
  }                                                                            \
  Tensor ret = shape(retShape);                                                \
  TensorIter it0(shape());                                                     \
  TensorIter it1(other.shape());                                               \
                                                                               \
  it0.broadcast(retShape);                                                     \
  it1.broadcast(retShape);                                                     \
                                                                               \
  for (int32_t idx = 0; idx < ret.elemCount_; idx++) {                         \
    ret[idx] = op((*this)[it0.next()], other[it1.next()]);                     \
  }                                                                            \
  return ret

#define TENSOR_UFUNC_FAST_LOOP(scalarRet, func)                                \
  if (t.isScalar()) {                                                          \
    return scalarRet;                                                          \
  }                                                                            \
  func functor;                                                                \
  functor.reset();                                                             \
  for (int32_t i = 0; i < t.elemCount_; i++) {                                 \
    functor.op(t.data_[i]);                                                    \
  }                                                                            \
  return functor.result()

#define TENSOR_UFUNC_REDUCE(func)                                              \
  func functor;                                                                \
  return t.reduce(functor, axis.get(t.dimCount_), keepDims)

Tensor Tensor::shape(const Shape &shape) {
  Tensor ret;
  ret.shape_ = shape;
  ret.initMeta();
  ret.initData();
  return ret;
}

Tensor Tensor::scalar(const float &value) {
  Tensor ret;
  ret.dimCount_ = 0;
  ret.elemCount_ = 1;
  ret.shape_.clear();
  ret.strides_.clear();
  ret.data_ = new float[1];
  ret.data_[0] = value;
  return ret;
}

Tensor Tensor::ones(const Shape &shape) {
  Tensor ret = Tensor::shape(shape);
  for (int32_t i = 0; i < ret.elemCount_; i++) {
    ret.data_[i] = 1.f;
  }
  return ret;
}

Tensor Tensor::onesLike(const Tensor &t) { return ones(t.shape()); }

Tensor Tensor::zeros(const Shape &shape) {
  Tensor ret = Tensor::shape(shape);
  memset(ret.data_, 0, ret.elemCount_ * sizeof(float));
  return ret;
}

Tensor Tensor::rand(const Shape &shape) {
  Tensor ret = Tensor::shape(shape);
  auto generator = RandomGenerator::getGenerator();
  std::uniform_real_distribution distribution(0.0f, 1.0f);
  for (int32_t i = 0; i < ret.elemCount_; i++) {
    ret.data_[i] = distribution(generator);
  }
  return ret;
}

Tensor Tensor::randn(const Shape &shape) {
  Tensor ret = Tensor::shape(shape);
  auto generator = RandomGenerator::getGenerator();
  std::normal_distribution distribution(0.0f, 1.0f);
  for (int32_t i = 0; i < ret.elemCount_; i++) {
    ret.data_[i] = distribution(generator);
  }
  return ret;
}

Tensor Tensor::bernoulli(const Shape &shape, float p) {
  Tensor ret = Tensor::shape(shape);
  auto generator = RandomGenerator::getGenerator();
  std::bernoulli_distribution distribution(p);
  for (int32_t i = 0; i < ret.elemCount_; i++) {
    ret.data_[i] = distribution(generator);
  }
  return ret;
}

Tensor Tensor::tri(int32_t n, int32_t m, int32_t k) {
  if (m <= 0) {
    m = n;
  }
  Tensor ret = shape({n, m});
  int32_t idx = 0;
  for (int32_t i = 0; i < n; i++) {
    for (int32_t j = 0; j < m; j++) {
      ret[idx++] = (j <= i + k) ? 1.f : 0.f;
    }
  }

  return ret;
}

Tensor::Tensor(const Array1d &values1d) {
  shape_ = {(int32_t)values1d.size()};
  initMeta();
  initData(values1d.data());
}

Tensor::Tensor(const Array2d &values2d) {
  shape_ = {(int32_t)values2d.size(), (int32_t)values2d[0].size()};
  initMeta();
  initData();
  for (int32_t idx = 0; idx < shape_[0]; idx++) {
    memcpy(data_ + idx * strides_[0], values2d[idx].data(),
           values2d[idx].size() * sizeof(float));
  }
}

Tensor::Tensor(const Array3d &values3d) {
  shape_ = {(int32_t)values3d.size(), (int32_t)values3d[0].size(),
            (int32_t)values3d[0][0].size()};
  initMeta();
  initData();
  for (int32_t idx = 0; idx < shape_[0]; idx++) {
    for (int32_t k = 0; k < shape_[1]; k++) {
      memcpy(data_ + idx * strides_[0] + k * strides_[1],
             values3d[idx][k].data(), values3d[idx][k].size() * sizeof(float));
    }
  }
}

void Tensor::initMeta() {
  dimCount_ = (int32_t)shape_.size();
  elemCount_ = 1;
  strides_.resize(dimCount_);
  for (auto dim = int32_t(dimCount_ - 1); dim >= 0; dim--) {
    strides_[dim] = elemCount_;
    elemCount_ *= shape_[dim];
  }
}

void Tensor::initData(const float *from) {
  data_ = new float[elemCount_];
  if (from) {
    memcpy(data_, from, elemCount_ * sizeof(float));
  }
}

Tensor Tensor::reshape(const std::vector<int32_t> &shape) {
  shape_.resize(shape.size());

  int32_t inferredIdx = -1;
  int32_t cnt = 1;
  for (int32_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      if (inferredIdx >= 0) {
        error(__FUNCTION__, TensorError_InvalidShape);
        return *this;
      }
      inferredIdx = i;
      shape_[i] = 0;
    } else {
      cnt *= shape[i];
      shape_[i] = shape[i];
    }
  }
  if (inferredIdx >= 0) {
    shape_[inferredIdx] = elemCount_ / cnt;
  }

  initMeta();
  return *this;
}

Tensor Tensor::reshape(const Tensor &t, const Shape &shape) {
  Tensor ret = t;
  ret.reshape(shape);
  return ret;
}

Tensor Tensor::reshape(const std::vector<int32_t> &shape) const {
  Tensor ret = *this;
  ret.reshape(shape);
  return ret;
}

void Tensor::flatten(int32_t startDim, int32_t endDim) {
  Shape retShape;
  for (int32_t i = 0; i < startDim; i++) {
    retShape.push_back(shape_[i]);
  }
  int32_t flattenDims = 1;
  if (endDim < 0) {
    endDim = dimCount_ - 1;
  }
  for (int32_t i = startDim; i <= endDim; i++) {
    flattenDims *= shape_[i];
  }
  retShape.push_back(flattenDims);
  for (int32_t i = endDim + 1; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }

  reshape(retShape);
}

Tensor Tensor::flatten(const Tensor &t, int32_t startDim, int32_t endDim) {
  Tensor ret = t;
  ret.flatten(startDim, endDim);
  return ret;
}

void Tensor::unflatten(int32_t dim, const std::vector<int32_t> &sizes) {
  if (dim < 0) {
    dim += dimCount_;
  }
  Shape retShape;
  for (int32_t i = 0; i < dim; i++) {
    retShape.push_back(shape_[i]);
  }
  int32_t unflattenDims = 1;
  int32_t inferredIdx = -1;
  for (int32_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == -1) {
      inferredIdx = dim + i;
      retShape.push_back(0);
    } else {
      unflattenDims *= sizes[i];
      retShape.push_back(sizes[i]);
    }
  }
  if (inferredIdx >= 0) {
    retShape[inferredIdx] = shape_[dim] / unflattenDims;
  } else if (unflattenDims != shape_[dim]) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return;
  }
  for (int32_t i = dim + 1; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  reshape(retShape);
}

Tensor Tensor::unflatten(const Tensor &t, int32_t dim,
                         const std::vector<int32_t> &sizes) {
  Tensor ret = t;
  ret.unflatten(dim, sizes);
  return ret;
}

void Tensor::squeeze(int32_t dim) {
  if (dim >= dimCount_) {
    return;
  }
  if (dim >= 0 && shape_[dim] != 1) {
    return;
  }
  Shape retShape;
  if (dim >= 0) {
    for (int32_t i = 0; i < dim; i++) {
      retShape.push_back(shape_[i]);
    }
    for (int32_t i = dim + 1; i < dimCount_; i++) {
      retShape.push_back(shape_[i]);
    }
  } else {
    for (auto d : shape_) {
      if (d != 1) {
        retShape.push_back(d);
      }
    }
  }
  reshape(retShape);
}

void Tensor::squeeze(const std::vector<int32_t> &dims) {
  for (const auto d : dims) {
    squeeze(d);
  }
}

Tensor Tensor::squeeze(const Tensor &t, int32_t dim) {
  Tensor ret = t;
  ret.squeeze(dim);
  return ret;
}

Tensor Tensor::squeeze(const Tensor &t, const std::vector<int32_t> &dims) {
  Tensor ret = t;
  ret.squeeze(dims);
  return ret;
}

void Tensor::unsqueeze(int32_t dim) {
  if (dim > dimCount_ || dim < -dimCount_ - 1) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return;
  }
  if (dim < 0) {
    dim += dimCount_ + 1;
  }
  Shape retShape;
  for (int32_t i = 0; i < dim; i++) {
    retShape.push_back(shape_[i]);
  }
  retShape.push_back(1);
  for (int32_t i = dim; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  reshape(retShape);
}

Tensor Tensor::unsqueeze(const Tensor &t, int32_t dim) {
  Tensor ret = t;
  ret.unsqueeze(dim);
  return ret;
}

void Tensor::fill(float value) {
  for (int32_t i = 0; i < elemCount_; i++) {
    data_[i] = value;
  }
}

Tensor Tensor::fill(const Tensor &t, float value) {
  Tensor ret = shape(t.shape());
  ret.fill(value);
  return ret;
}

void Tensor::clampMin(float min) {
  for (int32_t i = 0; i < elemCount_; i++) {
    data_[i] = std::max(data_[i], min);
  }
}

void Tensor::clampMax(float max) {
  for (int32_t i = 0; i < elemCount_; i++) {
    data_[i] = std::min(data_[i], max);
  }
}

void Tensor::clamp(float min, float max) {
  for (int32_t i = 0; i < elemCount_; i++) {
    data_[i] = std::max(min, std::min(data_[i], max));
  }
}

Tensor Tensor::clampMin(const Tensor &t, float min) {
  Tensor ret = t;
  ret.clampMin(min);
  return ret;
}

Tensor Tensor::clampMax(const Tensor &t, float max) {
  Tensor ret = t;
  ret.clampMax(max);
  return ret;
}

Tensor Tensor::clamp(const Tensor &t, float min, float max) {
  Tensor ret = t;
  ret.clamp(min, max);
  return ret;
}

std::vector<int32_t> Tensor::range(int32_t start, int32_t stop, int32_t step) {
  std::vector<int32_t> values;
  int32_t pos = start;
  while (pos < stop) {
    values.push_back(pos);
    pos += step;
  }
  return values;
}

Tensor Tensor::arange(float start, float stop, float step) {
  Array1d values;
  float pos = start;
  while (pos < stop) {
    values.push_back(pos);
    pos += step;
  }

  return Tensor(values);
}

Tensor Tensor::linspace(float start, float end, int steps) {
  Array1d values;
  if (steps > 0) {
    if (steps == 1) {
      values.push_back(start);
    } else {
      float step = (end - start) / ((float)steps - 1);
      for (int i = 0; i < steps; ++i) {
        values.push_back(start + (float)i * step);
      }
    }
  }
  return Tensor(values);
}

Tensor Tensor::indexInteger(const std::vector<int32_t> &idx,
                            float *dataPtr) const {
  auto len = (int32_t)idx.size();
  int32_t dataIdx = 0;
  for (int32_t i = 0; i < len; i++) {
    dataIdx += idx[i] * strides_[i];
  }
  int32_t dimStride = strides_[len - 1];
  if (dataPtr) {
    memcpy(dataPtr, &data_[dataIdx], dimStride * sizeof(float));
    return {};
  }
  Shape retShape;
  for (int32_t i = len; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  Tensor retTensor = shape(retShape);
  assert(dimStride == retTensor.size());
  memcpy(&retTensor[0], &data_[dataIdx], dimStride * sizeof(float));
  return retTensor;
}

Tensor Tensor::index(const std::vector<int32_t> &idx) const {
  return indexAdvance({idx});
}

Tensor
Tensor::indexAdvance(const std::vector<std::vector<int32_t>> &indexes) const {
  auto fistDim = (int32_t)indexes[0].size();
  auto dimStride = strides_[indexes.size() - 1];
  Shape retShape = {fistDim};
  for (auto i = (int32_t)indexes.size(); i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  Tensor retTensor = shape(retShape);
  std::vector<int32_t> idx;
  idx.resize(indexes.size());
  for (int32_t i = 0; i < fistDim; i++) {
    for (int32_t j = 0; j < indexes.size(); j++) {
      auto ind = indexes[j][i];
      idx[j] = ind >= 0 ? ind : ind + shape()[j];
    }
    indexInteger(idx, &retTensor[dimStride * i]);
  }

  return retTensor;
}

void Tensor::indexIntegerSet(const std::vector<int32_t> &idx, float val) {
  auto len = idx.size();
  int32_t dataIdx = 0;
  for (int32_t i = 0; i < len; i++) {
    dataIdx += idx[i] * strides_[i];
  }
  int32_t dimStride = strides_[len - 1];
  for (int32_t i = 0; i < dimStride; i++) {
    data_[dataIdx + i] = val;
  }
}

void Tensor::indexIntegerSet(const std::vector<int32_t> &idx,
                             const float *valPtr) {
  auto len = idx.size();
  int32_t dataIdx = 0;
  for (int32_t i = 0; i < len; i++) {
    dataIdx += idx[i] * strides_[i];
  }
  int32_t dimStride = strides_[len - 1];
  memcpy(&data_[dataIdx], valPtr, dimStride * sizeof(float));
}

void Tensor::indexIntegerSet(const std::vector<int32_t> &idx,
                             const Tensor &val) {
  indexIntegerSet(idx, &val.data_[0]);
}

void Tensor::indexAdvanceSet(const std::vector<std::vector<int32_t>> &indexes,
                             float val) {
  auto fistDim = (int32_t)indexes[0].size();
  Shape retShape = {fistDim};
  for (auto i = (int32_t)indexes.size(); i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  std::vector<int32_t> idx;
  idx.resize(indexes.size());
  for (int32_t i = 0; i < fistDim; i++) {
    for (int32_t j = 0; j < indexes.size(); j++) {
      auto ind = indexes[j][i];
      idx[j] = ind >= 0 ? ind : ind + shape()[j];
    }
    indexIntegerSet(idx, val);
  }
}

void Tensor::indexAdvanceSet(const std::vector<std::vector<int32_t>> &indexes,
                             const Tensor &val) {
  auto fistDim = (int32_t)indexes[0].size();
  Shape retShape = {fistDim};
  for (auto i = (int32_t)indexes.size(); i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  std::vector<int32_t> idx;
  idx.resize(indexes.size());

  int32_t valStride = strides_[idx.size() - 1];
  int32_t dataIdx = 0;
  for (int32_t i = 0; i < fistDim; i++) {
    for (int32_t j = 0; j < indexes.size(); j++) {
      auto ind = indexes[j][i];
      idx[j] = ind >= 0 ? ind : ind + shape()[j];
    }
    indexIntegerSet(idx, &val.data_[dataIdx]);
    dataIdx += valStride;
  }
}

Tensor Tensor::im2col(Size2D kernelSize, Size2D stride, Size2D padding) const {
  // this: [C, H, W], [N, C, H, W]
  assert(dimCount_ == 3 || dimCount_ == 4);
  int32_t batch = (dimCount_ == 4) ? shape_[0] : 1;
  int32_t channels = (dimCount_ == 4) ? shape_[1] : shape_[0];
  int32_t height = (dimCount_ == 4) ? shape_[2] : shape_[1];
  int32_t width = (dimCount_ == 4) ? shape_[3] : shape_[2];
  int32_t outH = (height - kernelSize.h + 2 * padding.h) / stride.h + 1;
  int32_t outW = (width - kernelSize.w + 2 * padding.w) / stride.w + 1;

  int32_t colH = outH * outW;
  int32_t colW = channels * kernelSize.h * kernelSize.w;
  auto retTensor =
      shape({batch, channels, kernelSize.h, kernelSize.w, outH, outW});

  int32_t imStride = strides_[0];
  int32_t colStride = colH * colW;
  for (int32_t n = 0; n < batch; n++) {
    for (int32_t c = 0; c < colW; ++c) {
      int32_t wOffset = c % kernelSize.w;
      int32_t hOffset = (c / kernelSize.h) % kernelSize.h;
      int32_t imC = c / (kernelSize.h * kernelSize.w);
      for (int32_t h = 0; h < outH; ++h) {
        for (int32_t w = 0; w < outW; ++w) {
          int32_t imRow = hOffset + h * stride.h;
          int32_t imCol = wOffset + w * stride.w;
          imRow -= padding.h;
          imCol -= padding.w;
          int32_t colIdx = (c * outH + h) * outW + w;
          int32_t imgIdx = imCol + width * (imRow + height * imC);
          if (imRow < 0 || imRow >= height || imCol < 0 || imCol >= width) {
            retTensor[n * colStride + colIdx] = 0; // zero padding
          } else {
            retTensor[n * colStride + colIdx] = data_[n * imStride + imgIdx];
          }
        }
      }
    }
  }
  return retTensor.transpose({0, 4, 5, 1, 2, 3})
      .reshape({batch * outH * outW, -1});
}

Tensor Tensor::col2im(const Shape &inputShape, Size2D kernelSize, Size2D stride,
                      Size2D padding) const {
  // inputShape: [C, H, W], [N, C, H, W]
  assert(inputShape.size() == 3 || inputShape.size() == 4);
  int32_t batch = (inputShape.size() == 4) ? inputShape[0] : 1;
  int32_t channels = (inputShape.size() == 4) ? inputShape[1] : inputShape[0];
  int32_t height = (inputShape.size() == 4) ? inputShape[2] : inputShape[1];
  int32_t width = (inputShape.size() == 4) ? inputShape[3] : inputShape[2];

  auto outH = (height - kernelSize.h + 2 * padding.h) / stride.h + 1;
  auto outW = (width - kernelSize.w + 2 * padding.w) / stride.w + 1;

  int32_t colH = outH * outW;
  int32_t colW = channels * kernelSize.h * kernelSize.w;

  auto col = reshape({batch, outH, outW, channels, kernelSize.h, kernelSize.w});
  col = col.transpose({0, 3, 4, 5, 1, 2});
  Tensor retTensor = zeros(inputShape);

  auto imStride = retTensor.strides_[0];
  auto colStride = colH * colW;
  for (int32_t n = 0; n < batch; n++) {
    for (int32_t c = 0; c < colW; ++c) {
      int32_t wOffset = c % kernelSize.w;
      int32_t hOffset = (c / kernelSize.h) % kernelSize.h;
      int32_t imC = c / (kernelSize.h * kernelSize.w);
      for (int32_t h = 0; h < outH; ++h) {
        for (int32_t w = 0; w < outW; ++w) {
          int32_t imRow = hOffset + h * stride.h;
          int32_t imCol = wOffset + w * stride.w;
          imRow -= padding.h;
          imCol -= padding.w;
          int32_t colIdx = (c * outH + h) * outW + w;
          int32_t imgIdx = imCol + width * (imRow + height * imC);
          if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
            retTensor[n * imStride + imgIdx] +=
                col.data_[n * colStride + colIdx];
          }
        }
      }
    }
  }
  return retTensor;
}

Tensor Tensor::transpose(const std::vector<int32_t> &axis) const {
  TENSOR_CHECK_EMPTY(*this, {});
  if (dim() <= 1) {
    return *this;
  }

  TensorIter it(shape());
  if (axis.empty()) {
    // If not specified, defaults to range(a.ndim)[::-1], which reverses the
    // order of the axes.
    std::vector<int32_t> reverseTrans;
    reverseTrans.resize(dim());
    for (int32_t i = 0; i < dim(); i++) {
      reverseTrans[i] = dim() - i - 1;
    }
    it.transpose(reverseTrans);
  } else if (axis.size() != dim()) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  } else {
    it.transpose(axis);
  }

  Tensor ret = shape(it.shape());
  for (int32_t idx = 0; idx < ret.elemCount_; idx++) {
    ret[idx] = data_[it.next()];
  }
  return ret;
}

std::vector<Tensor> Tensor::split(int32_t sections, const Axis &axis) const {
  int32_t axisDim = axis.get(dim());
  if (axisDim >= dim()) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  int32_t dimSize = shape()[axisDim];
  if (dimSize % sections != 0) {
    error(__FUNCTION__, TensorError_InvalidSections);
    return {};
  }

  // index of result tensors
  std::vector<int32_t> splitIndices;
  splitIndices.resize(dimSize);

  int32_t splitStride = dimSize / sections;
  int32_t splitIdx = 0;
  int32_t idx = 0;
  for (int32_t i = 0; i < dimSize; i++) {
    if (idx >= splitStride) {
      splitIdx++;
      idx = 0;
    }
    idx++;
    splitIndices[i] = splitIdx;
  }

  std::vector<Tensor> retTensors;
  retTensors.resize(sections);

  // init shape of result tensors
  Shape retShape = shape();
  retShape[axisDim] = splitStride;
  for (int32_t i = 0; i < sections; i++) {
    retTensors[i] = shape(retShape);
  }

  // do split
  splitAxis(retTensors, splitIndices, axisDim);
  return retTensors;
}

std::vector<Tensor> Tensor::split(const std::vector<int32_t> &indices,
                                  const Axis &axis) const {
  if (indices.empty()) {
    error(__FUNCTION__, TensorError_InvalidSections);
    return {};
  }

  int32_t axisDim = axis.get(dim());
  if (axisDim >= dim()) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // index of result tensors
  std::vector<int32_t> splitIndices;
  int32_t dimSize = shape()[axisDim];
  splitIndices.resize(dimSize);

  int32_t splitIdx = 0;
  int32_t idx = 0;
  for (int32_t i = 0; i < dimSize; i++) {
    if (splitIdx < indices.size() && idx >= indices[splitIdx]) {
      splitIdx++;
      idx = 0;
    }
    idx++;
    splitIndices[i] = splitIdx;
  }

  std::vector<Tensor> retTensors;
  retTensors.resize(indices.size() + 1);

  // init shape of result tensors
  Shape retShape = shape();
  // first section
  retShape[axisDim] = (int32_t)indices[0];
  retTensors[0] = shape(retShape);
  // middle sections
  for (int32_t i = 1; i < indices.size(); i++) {
    retShape[axisDim] = (int32_t)indices[i] - (int32_t)indices[i - 1];
    retTensors[i] = shape(retShape);
  }
  // last section
  retShape[axisDim] = dimSize - (int32_t)indices.back();
  retTensors[indices.size()] = shape(retShape);

  // do split
  splitAxis(retTensors, splitIndices, axisDim);
  return retTensors;
}

Tensor
Tensor::concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  int32_t totalSize = 0;
  for (auto &t : arrays) {
    totalSize += t.get().size();
  }

  Tensor retTensor = shape({totalSize});
  int32_t idx = 0;
  for (auto &t : arrays) {
    memcpy(&retTensor[idx], &t.get()[0], t.get().size() * sizeof(float));
    idx += t.get().size();
  }
  return retTensor;
}

Tensor
Tensor::concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays,
                    const Axis &axis) {
  // check axis
  auto &t0 = arrays[0].get();
  int32_t axisDim = axis.get(t0.dim());
  if (axisDim >= t0.dim()) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // check shapes
  if (!checkShapeEqual(arrays, axisDim)) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // init concat tensor idx
  std::vector<int32_t> concatIndices;
  for (int32_t i = 0; i < arrays.size(); i++) {
    int32_t dim = arrays[i].get().shape()[axisDim];
    for (int32_t j = 0; j < dim; j++) {
      concatIndices.emplace_back(i);
    }
  }

  Shape retShape = t0.shape();
  retShape[axisDim] = (int32_t)concatIndices.size();

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor Tensor::stack(const std::vector<std::reference_wrapper<Tensor>> &arrays,
                     const Axis &axis) {
  // check axis
  auto &t0 = arrays[0].get();
  int32_t axisDim = axis.get(t0.dim() + 1);
  if (axisDim > t0.dim()) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // check shapes
  for (int32_t i = 1; i < arrays.size(); i++) {
    auto &t = arrays[i].get();
    if (t.shape() != t0.shape()) {
      error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
  }

  // init result shape
  Shape retShape = t0.shape();
  retShape.insert(retShape.begin() + axisDim, (int32_t)arrays.size());
  Tensor retTensor = shape(retShape);

  // do stack
  std::vector<int32_t> srcIdx;
  srcIdx.resize(arrays.size());
  std::fill(srcIdx.begin(), srcIdx.end(), 0);

  int32_t dstIdx = 0;
  TensorIter it(retShape);
  while ((dstIdx = it.next()) >= 0) {
    int32_t sectionId = it.coordinates()[axisDim];
    retTensor[dstIdx] = arrays[sectionId].get()[srcIdx[sectionId]++];
  }

  return retTensor;
}

Tensor
Tensor::vstack(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  auto &t0 = arrays[0].get();
  int32_t axisDim = 0;

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() == 1) {
    // 1-D arrays must have the same length
    for (int32_t i = 1; i < arrays.size(); i++) {
      auto &t = arrays[i].get();
      if (t.shape() != t0.shape()) {
        shapesAligned = false;
        break;
      }
    }
  } else {
    shapesAligned = checkShapeEqual(arrays, axisDim);
  }
  if (!shapesAligned) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // init concat tensor idx
  std::vector<int32_t> concatIndices;
  if (t0.dim() == 1) {
    for (int32_t i = 0; i < arrays.size(); i++) {
      concatIndices.emplace_back(i);
    }
  } else {
    for (int32_t i = 0; i < arrays.size(); i++) {
      int32_t dim = arrays[i].get().shape()[axisDim];
      for (int32_t j = 0; j < dim; j++) {
        concatIndices.emplace_back(i);
      }
    }
  }

  Shape retShape;
  if (t0.dim() == 1) {
    retShape = {(int32_t)concatIndices.size(), t0.shape()[0]};
  } else {
    retShape = t0.shape();
    retShape[axisDim] = (int32_t)concatIndices.size();
  }

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor
Tensor::hstack(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  auto &t0 = arrays[0].get();
  int32_t axisDim = 1;

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() == 1) {
    // 1-D arrays which can be any length
    for (int32_t i = 1; i < arrays.size(); i++) {
      auto &t = arrays[i].get();
      if (t.dim() != t0.dim()) {
        shapesAligned = false;
        break;
      }
    }
  } else {
    shapesAligned = checkShapeEqual(arrays, axisDim);
  }
  if (!shapesAligned) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // This is equivalent to concatenation along the second axis,
  // except for 1-D arrays where it concatenates along the first axis
  if (t0.dim() == 1) {
    return concatenate(arrays);
  }

  // init concat tensor idx
  std::vector<int32_t> concatIndices;
  for (int32_t i = 0; i < arrays.size(); i++) {
    int32_t dim = arrays[i].get().shape()[axisDim];
    for (int32_t j = 0; j < dim; j++) {
      concatIndices.emplace_back(i);
    }
  }

  Shape retShape = t0.shape();
  retShape[axisDim] = (int32_t)concatIndices.size();

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor
Tensor::dstack(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  auto &t0 = arrays[0].get();
  int32_t axisDim = 2;

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() <= 2) {
    // 1-D or 2-D arrays must have the same shape
    for (int32_t i = 1; i < arrays.size(); i++) {
      auto &t = arrays[i].get();
      if (t.shape() != t0.shape()) {
        shapesAligned = false;
        break;
      }
    }
  } else {
    shapesAligned = checkShapeEqual(arrays, axisDim);
  }
  if (!shapesAligned) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // init concat tensor idx
  std::vector<int32_t> concatIndices;
  if (t0.dim() <= 2) {
    for (int32_t i = 0; i < arrays.size(); i++) {
      concatIndices.emplace_back(i);
    }
  } else {
    for (int32_t i = 0; i < arrays.size(); i++) {
      int32_t dim = arrays[i].get().shape()[axisDim];
      for (int32_t j = 0; j < dim; j++) {
        concatIndices.emplace_back(i);
      }
    }
  }

  Shape retShape;
  if (t0.dim() == 1) {
    retShape = {1, t0.shape()[0], (int32_t)concatIndices.size()};
  } else if (t0.dim() == 2) {
    retShape = {t0.shape()[0], t0.shape()[1], (int32_t)concatIndices.size()};
  } else {
    retShape = t0.shape();
    retShape[axisDim] = (int32_t)concatIndices.size();
  }

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor Tensor::operator<(const Tensor &other) const {
  TENSOR_CHECK_SHAPE_EQUAL_RET({});
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] < other[idx] ? 1.f : 0.f);
}

Tensor Tensor::operator>(const Tensor &other) const {
  TENSOR_CHECK_SHAPE_EQUAL_RET({});
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] > other[idx] ? 1.f : 0.f);
}

Tensor Tensor::operator==(const Tensor &other) const {
  TENSOR_CHECK_SHAPE_EQUAL_RET({});
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] == other[idx] ? 1.f : 0.f);
}

Tensor Tensor::operator!=(const Tensor &other) const {
  TENSOR_CHECK_SHAPE_EQUAL_RET({});
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] != other[idx] ? 1.f : 0.f);
}

Tensor Tensor::operator<(const float &other) const {
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] < other ? 1.f : 0.f);
}

Tensor Tensor::operator>(const float &other) const {
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] > other ? 1.f : 0.f);
}

Tensor Tensor::operator==(const float &other) const {
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] == other ? 1.f : 0.f);
}

Tensor Tensor::operator!=(const float &other) const {
  TENSOR_MATH_FAST_LOOP_PAIR(shape(shape()), =,
                             (*this)[idx] != other ? 1.f : 0.f);
}

Tensor Tensor::operator+(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {});
  TENSOR_MATH_BROADCAST_PAIR(+);
}

Tensor Tensor::operator-(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {});
  TENSOR_MATH_BROADCAST_PAIR(-);
}

Tensor Tensor::operator*(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {});
  TENSOR_MATH_BROADCAST_PAIR(*);
}

Tensor Tensor::operator/(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {});
  TENSOR_MATH_BROADCAST_PAIR(/);
}

void Tensor::operator+=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, );
  TENSOR_CHECK_SHAPE_EQUAL;
  TENSOR_MATH_FAST_LOOP_SELF(+=, other[idx]);
}

void Tensor::operator-=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, );
  TENSOR_CHECK_SHAPE_EQUAL;
  TENSOR_MATH_FAST_LOOP_SELF(-=, other[idx]);
}

void Tensor::operator*=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, );
  TENSOR_CHECK_SHAPE_EQUAL;
  TENSOR_MATH_FAST_LOOP_SELF(*=, other[idx]);
}

void Tensor::operator/=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, );
  TENSOR_CHECK_SHAPE_EQUAL;
  TENSOR_MATH_FAST_LOOP_SELF(/=, other[idx]);
}

Tensor Tensor::operator+(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {});
  TENSOR_MATH_FAST_LOOP_PAIR(*this, +=, other);
}

Tensor Tensor::operator-(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {});
  TENSOR_MATH_FAST_LOOP_PAIR(*this, -=, other);
}

Tensor Tensor::operator*(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {});
  TENSOR_MATH_FAST_LOOP_PAIR(*this, *=, other);
}

Tensor Tensor::operator/(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {});
  TENSOR_MATH_FAST_LOOP_PAIR(*this, /=, other);
}

void Tensor::operator+=(const float &other) {
  TENSOR_CHECK_EMPTY(*this, );
  TENSOR_MATH_FAST_LOOP_SELF(+=, other);
}

void Tensor::operator-=(const float &other) {
  TENSOR_CHECK_EMPTY(*this, );
  TENSOR_MATH_FAST_LOOP_SELF(-=, other);
}

void Tensor::operator*=(const float &other) {
  TENSOR_CHECK_EMPTY(*this, );
  TENSOR_MATH_FAST_LOOP_SELF(*=, other);
}

void Tensor::operator/=(const float &other) {
  TENSOR_CHECK_EMPTY(*this, );
  TENSOR_MATH_FAST_LOOP_SELF(/=, other);
}

Tensor operator+(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {});
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other + ret[idx]);
}

Tensor operator-(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {});
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other - ret[idx]);
}

Tensor operator*(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {});
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other * ret[idx]);
}

Tensor operator/(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {});
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other / ret[idx]);
}

Tensor Tensor::sin(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, std::sin(ret[idx]));
}

Tensor Tensor::cos(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, std::cos(ret[idx]));
}

Tensor Tensor::sqrt(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, std::sqrt(ret[idx]));
}

Tensor Tensor::tanh(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, fastTanh(ret[idx]));
}

Tensor Tensor::exp(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, std::exp(ret[idx]));
}

Tensor Tensor::log(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, std::log(ret[idx]));
}

Tensor Tensor::pow(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {});
  if (other.isScalar()) {
    return this->pow(other[0]);
  }
  TENSOR_MATH_BROADCAST_PAIR_FUNC(std::pow);
}

Tensor Tensor::pow(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {});
  TENSOR_MATH_FAST_LOOP_PAIR(*this, =, std::pow(ret[idx], other));
}

float Tensor::dot(const float &a, const float &b) { return a * b; }

Tensor Tensor::dot(const Tensor &a, const float &b) {
  TENSOR_CHECK_EMPTY(a, {});
  return a * b;
}

Tensor Tensor::dot(const float &a, const Tensor &b) {
  TENSOR_CHECK_EMPTY(b, {});
  return b * a;
}

Tensor Tensor::dot(const Tensor &a, const Tensor &b) {
  TENSOR_CHECK_EMPTY_PAIR(a, b, {});

  // If both a and b are 1-D arrays, it is inner product of vectors
  if (a.dim() == 1 && b.dim() == 1) {
    if (a.size() != b.size()) {
      error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
    float ret = 0.f;
    for (int32_t i = 0; i < a.size(); i++) {
      ret += a[i] * b[i];
    }
    return Tensor::scalar(ret);
  }

  // If both a and b are 2-D arrays, it is matrix multiplication
  if (a.dim() == 2 && b.dim() == 2) {
    int32_t m = a.shape()[0];
    int32_t middle = a.shape()[1];
    if (middle != b.shape()[0]) {
      error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
    int32_t n = b.shape()[1];
    Tensor ret = shape({m, n});
    Blas::gemm(&ret[0], &a[0], &b[0], (int)m, (int)middle, (int)n);
    return ret;
  }

  // If either a or b is 0-D (scalar), it is equivalent to multiply
  if (a.isScalar()) {
    return b * a[0];
  }
  if (b.isScalar()) {
    return a * b[0];
  }

  // If a is an N-D array and b is a 1-D array, it is a sum product over the
  // last axis of a and b. If a is an N-D array and b is an M-D array (where
  // M>=2), it is a sum product over the last axis of a and the second-to-last
  // axis of b:
  error(__FUNCTION__, TensorError_NotSupport);
  return {};
}

Tensor Tensor::matmul(const Tensor &a, const Tensor &b) {
  TENSOR_CHECK_EMPTY_PAIR(a, b, {});

  // Multiplication by scalars is not allowed, use * instead.
  if (a.isScalar() || b.isScalar()) {
    error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }

  // rules:
  // If both arguments are 2-D they are multiplied like conventional matrices.
  // If the first argument is 1-D, it is promoted to a matrix by prepending a 1
  // to its dimensions. After matrix multiplication the prepended 1 is removed.
  // If the second argument is 1-D, it is promoted to a matrix by appending a 1
  // to its dimensions. After matrix multiplication the appended 1 is removed.

  Shape shapeA = a.shape();
  Shape shapeB = b.shape();
  bool prependA = false;
  bool appendB = false;
  if (shapeA.size() == 1) {
    shapeA.insert(shapeA.begin(), 1);
    prependA = true;
  }
  if (shapeB.size() == 1) {
    shapeB.insert(shapeB.end(), 1);
    appendB = true;
  }

  // check matrix multiplication compatible
  if (shapeA.back() != shapeB[shapeB.size() - 2]) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // check shape broadcast compatible
  Shape retShape;
  auto compatible = checkCompatible(shapeA, shapeB, retShape, 2);
  if (compatible == ShapeCompatible_Error) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  auto retDimCnt = (int32_t)retShape.size();
  auto m = shapeA[shapeA.size() - 2];
  auto k = shapeA.back();
  auto n = shapeB.back();

  retShape[retDimCnt - 2] = m;
  retShape[retDimCnt - 1] = n;
  Tensor retTensor = shape(retShape);
  if (retDimCnt > 2) {
    // broadcast matrix multiplication
    auto broadcastShape = Shape(retShape.begin(), retShape.end() - 2);
    TensorIter iterRet(broadcastShape);
    TensorIter iterA = shapeA.size() > 2
                           ? TensorIter({shapeA.begin(), shapeA.end() - 2})
                           : TensorIter({1});
    TensorIter iterB = shapeB.size() > 2
                           ? TensorIter({shapeB.begin(), shapeB.end() - 2})
                           : TensorIter({1});
    iterA.broadcast(broadcastShape);
    iterB.broadcast(broadcastShape);
    auto strideRet = m * n;
    auto strideA = m * k;
    auto strideB = k * n;
    for (int32_t idx = 0; idx < iterRet.size(); idx++) {
      Blas::gemm(&retTensor[iterRet.next() * strideRet],
                 &a[iterA.next() * strideA], &b[iterB.next() * strideB], (int)m,
                 (int)k, (int)n);
    }
  } else {
    Blas::gemm(&retTensor[0], &a[0], &b[0], (int)m, (int)k, (int)n);
    if (prependA) {
      retTensor.reshape({n});
    }
  }

  // reduce dimension if necessary
  if (appendB) {
    if (prependA) {
      retTensor = Tensor::scalar(retTensor[0]);
    } else {
      retTensor.reshape({m});
    }
  }

  return retTensor;
}

Tensor Tensor::matmulTrans(const Tensor &a, const Tensor &b) {
  // fast path
  if (a.dim() == 2 && b.dim() == 2) {
    // a[m, k], b[n, k] -> [k, n]
    int32_t m = a.shape()[0];
    int32_t k = a.shape()[1];
    int32_t n = b.shape()[0];
    if (k != b.shape()[1]) {
      error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
    Tensor retTensor = shape({m, n});
    Blas::gemmTrans(&retTensor[0], &a[0], &b[0], (int)m, (int)k, (int)n);
    return retTensor;
  }

  // slow path
  return matmul(a, b.transpose());
}

float Tensor::min(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0);
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncMin);
}

float Tensor::max(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0);
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncMax);
}

float Tensor::mean(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0);
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncMean);
}

float Tensor::sum(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0);
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncSum);
}

float Tensor::var(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0);
  TENSOR_UFUNC_FAST_LOOP(0, UFuncVar);
}

float Tensor::argmin(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0);
  TENSOR_UFUNC_FAST_LOOP(0, UFuncArgMin);
}

float Tensor::argmax(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0);
  TENSOR_UFUNC_FAST_LOOP(0, UFuncArgMax);
}

Tensor Tensor::min(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_UFUNC_REDUCE(UFuncMin);
}

Tensor Tensor::max(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_UFUNC_REDUCE(UFuncMax);
}

Tensor Tensor::mean(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_UFUNC_REDUCE(UFuncMean);
}

Tensor Tensor::sum(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_UFUNC_REDUCE(UFuncSum);
}

Tensor Tensor::var(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_UFUNC_REDUCE(UFuncVar);
}

Tensor Tensor::argmin(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_UFUNC_REDUCE(UFuncArgMin);
}

Tensor Tensor::argmax(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {});
  TENSOR_UFUNC_REDUCE(UFuncArgMax);
}

void Tensor::traverse(UFunc &func, int32_t start, int32_t stride,
                      int32_t cnt) const {
  int32_t idx = start;
  for (int32_t n = 0; n < cnt; n++) {
    func.op(data_[idx]);
    idx += stride;
  }
}

Tensor Tensor::reduce(UFunc &func, int32_t axis, bool keepDims) const {
  // check scalar
  if (isScalar()) {
    return scalar(data_[0]);
  }

  // check axis
  if (axis >= dimCount_) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // construct result shape
  Shape retShape;
  retShape.reserve(dimCount_);
  for (int32_t dim = 0; dim < dimCount_; dim++) {
    if (axis == dim) {
      if (keepDims) {
        retShape.emplace_back(1);
      }
    } else {
      retShape.emplace_back(shape_[dim]);
    }
  }

  // reduce via function
  Tensor ret = shape(retShape);

  int32_t axisStride = strides_[axis];
  int32_t axisLength = shape_[axis];

  int32_t groupStride = axisStride * axisLength;
  int32_t groupCount = elemCount_ / groupStride;

  int32_t retIdx = 0;
  int32_t axisStart = 0;
  for (int32_t i = 0; i < groupCount; i++) {
    axisStart = i * groupStride;
    for (int32_t j = 0; j < axisStride; j++) {
      func.reset();
      traverse(func, axisStart, axisStride, axisLength);
      ret[retIdx++] = func.result();
      axisStart++;
    }
  }

  return ret;
}

void Tensor::splitAxis(std::vector<Tensor> &retTensors,
                       std::vector<int32_t> &splitIndices, int32_t axis) const {
  std::vector<int32_t> dstIdx;
  dstIdx.resize(retTensors.size());
  std::fill(dstIdx.begin(), dstIdx.end(), 0);

  TensorIter it(shape());
  int32_t srcIdx = 0;
  while ((srcIdx = it.next()) >= 0) {
    int32_t sectionId = splitIndices[it.coordinates()[axis]];
    retTensors[sectionId][dstIdx[sectionId]++] = data_[srcIdx];
  }
}

Tensor
Tensor::arraysConcat(const std::vector<std::reference_wrapper<Tensor>> &arrays,
                     const Shape &retShape,
                     const std::vector<int32_t> &concatIndices, int32_t axis) {
  Tensor retTensor = shape(retShape);

  // do concat
  std::vector<int32_t> srcIdx;
  srcIdx.resize(arrays.size());
  std::fill(srcIdx.begin(), srcIdx.end(), 0);

  int32_t dstIdx = 0;
  TensorIter it(retShape);
  while ((dstIdx = it.next()) >= 0) {
    int32_t sectionId = concatIndices[it.coordinates()[axis]];
    retTensor[dstIdx] = arrays[sectionId].get()[srcIdx[sectionId]++];
  }

  return retTensor;
}

ShapeCompatible Tensor::checkCompatible(const Shape &t0, const Shape &t1,
                                        Shape &retShape, int32_t skipLast) {
  retShape = t0.size() > t1.size() ? t0 : t1;

  auto idxRet = (int32_t)(retShape.size() - 1 - skipLast);
  auto idx0 = (int32_t)(t0.size() - 1 - skipLast);
  auto idx1 = (int32_t)(t1.size() - 1 - skipLast);

  bool needBroadcast = false; // dimensions already exist
  while (idx0 >= 0 && idx1 >= 0) {
    auto dim0 = t0[idx0];
    auto dim1 = t1[idx1];
    if (dim0 != dim1) {
      if (dim0 == 1 || dim1 == 1) {
        retShape[idxRet] = std::max(dim0, dim1);
        needBroadcast = true;
      } else {
        return ShapeCompatible_Error;
      }
    }

    idxRet--;
    idx0--;
    idx1--;
  }

  if (!needBroadcast && t0.size() == t1.size()) {
    return ShapeCompatible_SameShape;
  }

  return ShapeCompatible_Broadcast;
}

bool Tensor::checkShapeEqual(
    const std::vector<std::reference_wrapper<Tensor>> &arrays,
    int32_t exceptAxis) {
  auto &t0 = arrays[0].get();

  for (int32_t i = 1; i < arrays.size(); i++) {
    auto &t = arrays[i].get();
    if (t.dim() != t0.dim()) {
      return false;
    }
    for (int32_t j = 0; j < t.dim(); j++) {
      if (j != exceptAxis && t.shape_[j] != t0.shape_[j]) {
        return false;
      }
    }
  }

  return true;
}

void Tensor::error(const char *where, TensorError error) {
  switch (error) {
  case TensorError_EmptyTensor:
    LOGE("[%s] Tensor error: empty tensor", where);
    break;
  case TensorError_InvalidShape:
    LOGE("[%s] Tensor error: invalid shape", where);
    break;
  case TensorError_InvalidAxis:
    LOGE("[%s] Tensor error: invalid axis", where);
    break;
  case TensorError_InvalidSections:
    LOGE("[%s] Tensor error: invalid sections", where);
    break;
  case TensorError_ShapeNotAligned:
    LOGE("[%s] Tensor error: shapes not aligned", where);
    break;
  case TensorError_NotSupport:
    LOGE("[%s] Tensor error: function not support", where);
    break;
  default:
    break;
  }

#ifdef DEBUG
  abort();
#endif
}

// Ref: https://math.stackexchange.com/a/446411
float Tensor::fastTanh(float x) {
  if (x < -4.97) {
    return -1.0;
  }
  if (x > 4.97) {
    return 1.0;
  }
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return a / b;
}

TensorIter::TensorIter(const Shape &shape) { reshape(shape); }

Shape TensorIter::shape() {
  Shape ret;
  ret.resize(ndM1_ + 1);
  for (int32_t i = 0; i <= ndM1_; i++) {
    ret[i] = dimsM1_[i] + 1;
  }
  return ret;
}

void TensorIter::reshape(const Shape &shape) {
  ndM1_ = (int32_t)shape.size() - 1;
  size_ = 1;
  for (auto dim = int32_t(ndM1_); dim >= 0; dim--) {
    dimsM1_[dim] = (int32_t)shape[dim] - 1;
    strides_[dim] = size_;
    backStrides_[dim] = strides_[dim] * dimsM1_[dim];

    size_ *= (int32_t)shape[dim];
  }

  reset();
}

int32_t TensorIter::next() {
  if (itCnt_ >= size_) {
    return -1;
  }

  if (index_ < 0) {
    index_ = 0;
  } else {
    for (auto dim = ndM1_; dim >= 0; dim--) {
      if (coordinates_[dim] < dimsM1_[dim]) {
        coordinates_[dim]++;
        index_ += strides_[dim];
        break;
      } else {
        coordinates_[dim] = 0;
        index_ -= backStrides_[dim];
      }
    }
  }

  itCnt_++;
  return index_;
}

void TensorIter::reset() {
  for (int32_t i = 0; i < ndM1_ + 1; i++) {
    coordinates_[i] = 0;
  }
  index_ = -1;
  itCnt_ = 0;
}

void TensorIter::broadcast(const Shape &shape) {
  int32_t targetNdM1_ = (int32_t)shape.size() - 1;

  // origin dimensions
  for (auto dim = ndM1_; dim >= 0; dim--) {
    int32_t targetDim = targetNdM1_ - ndM1_ + dim;
    if (dimsM1_[dim] != shape[targetDim] - 1) {
      // broadcast dimension, set stride & back stride to zero
      strides_[targetDim] = 0;
      backStrides_[targetDim] = 0;
    } else {
      strides_[targetDim] = strides_[dim];
      backStrides_[targetDim] = backStrides_[dim];
    }
  }

  // new dimensions
  for (int32_t dim_ = 0; dim_ < targetNdM1_ - ndM1_; dim_++) {
    strides_[dim_] = 0;
    backStrides_[dim_] = 0;
  }

  // update shape
  ndM1_ = (int32_t)shape.size() - 1;
  size_ = 1;
  for (auto dim = int32_t(ndM1_); dim >= 0; dim--) {
    dimsM1_[dim] = (int32_t)shape[dim] - 1;
    size_ *= (int32_t)shape[dim];
  }

  // reset
  reset();
}

void TensorIter::transpose(const std::vector<int32_t> &axis) {
  // assume axis size equal to dimension count
  assert(axis.size() == ndM1_ + 1);

  // reorder dimsM1_, strides_, backStrides_
  reorder(dimsM1_, axis);
  reorder(strides_, axis);
  reorder(backStrides_, axis);
}

} // namespace TinyGPT
