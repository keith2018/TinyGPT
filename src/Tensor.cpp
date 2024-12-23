/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor.h"
#include "Logger.h"
#include "Blas.h"

#include <cassert>
#include <cstring>
#include <cmath>

namespace TinyGPT {

#define TENSOR_CHECK_EMPTY(t, ret)                              \
  if ((t).empty()) {                                            \
    Tensor::error(__FUNCTION__, TensorError_EmptyTensor);       \
    return ret;                                                 \
  }                                                             \

#define TENSOR_CHECK_EMPTY_PAIR(t1, t2, ret)                    \
  if ((t1).empty() || (t2).empty()) {                           \
    Tensor::error(__FUNCTION__, TensorError_EmptyTensor);       \
    return ret;                                                 \
  }

#define TENSOR_MATH_FAST_LOOP_SELF(op, other)                   \
  for (uint32_t idx = 0; idx < elemCount_; idx++) {             \
    data_[idx] op other;                                        \
  }                                                             \

#define TENSOR_MATH_FAST_LOOP_PAIR(init, op, other)             \
  Tensor ret = init;                                            \
  for (uint32_t idx = 0; idx < ret.elemCount_; idx++) {         \
    ret[idx] op other;                                          \
  }                                                             \
  return ret;                                                   \

#define TENSOR_CHECK_SHAPE_EQUAL                                \
  if (shape_ != other.shape_) {                                 \
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);   \
    return;                                                     \
  }                                                             \

#define TENSOR_MATH_BROADCAST_LOOP(retShape, op)                \
  Tensor ret = Tensor::shape(retShape);                         \
  TensorIter it0(shape());                                      \
  TensorIter it1(other.shape());                                \
                                                                \
  it0.broadcast(retShape);                                      \
  it1.broadcast(retShape);                                      \
                                                                \
  for (uint32_t idx = 0; idx < ret.elemCount_; idx++) {         \
    ret[idx] = (*this)[it0.next()] op other[it1.next()];        \
  }                                                             \
                                                                \
  return ret;                                                   \

#define TENSOR_MATH_BROADCAST_PAIR(op)                          \
  if (this->isScalar()) {                                       \
    return other op (*this)[0];                                 \
  }                                                             \
  if (other.isScalar()) {                                       \
    return *this op other[0];                                   \
  }                                                             \
  Shape retShape;                                               \
  auto comp = checkCompatible(shape(), other.shape(), retShape);\
  if (comp == ShapeCompatible_Error) {                          \
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);   \
    return {};                                                  \
  }                                                             \
  if (comp == ShapeCompatible_SameShape) {                      \
    TENSOR_MATH_FAST_LOOP_PAIR(*this, op##=, other[idx])        \
  }                                                             \
  TENSOR_MATH_BROADCAST_LOOP(retShape, op)                      \

#define TENSOR_UFUNC_FAST_LOOP(scalarRet, func)                 \
  if (t.isScalar()) {                                           \
    return scalarRet;                                           \
  }                                                             \
  func functor;                                                 \
  functor.reset();                                              \
  for (auto &v : t.data_) {                                     \
    functor.op(v);                                              \
  }                                                             \
  return functor.result();                                      \

#define TENSOR_UFUNC_REDUCE(func)                               \
  func functor;                                                 \
  return t.reduce(functor, axis.get(t.dimCount_), keepDims);    \


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
  ret.data_.resize(1);
  ret.data_[0] = value;
  return ret;
}

Tensor Tensor::ones(const Shape &shape) {
  Tensor ret = Tensor::shape(shape);
  std::fill(ret.data_.begin(), ret.data_.end(), 1.f);
  return ret;
}

Tensor Tensor::zeros(const Shape &shape) {
  Tensor ret = Tensor::shape(shape);
  std::fill(ret.data_.begin(), ret.data_.end(), 0.f);
  return ret;
}

Tensor Tensor::tri(uint32_t n, uint32_t m, int32_t k) {
  if (m <= 0) { m = n; }
  Tensor ret = Tensor::shape({n, m});
  uint32_t idx = 0;
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < m; j++) {
      ret[idx++] = ((int32_t) j <= (int32_t) i + k) ? 1.f : 0.f;
    }
  }

  return ret;
}

Tensor::Tensor(const Array1d &values1d) {
  shape_ = {(uint32_t) values1d.size()};
  initMeta();
  data_ = values1d;
}

Tensor::Tensor(const Array2d &values2d) {
  shape_ = {(uint32_t) values2d.size(), (uint32_t) values2d[0].size()};
  initMeta();
  for (uint32_t idx = 0; idx < shape_[0]; idx++) {
    data_.insert(data_.end(), values2d[idx].begin(), values2d[idx].end());
  }
}

Tensor::Tensor(const Array3d &values3d) {
  shape_ = {(uint32_t) values3d.size(), (uint32_t) values3d[0].size(), (uint32_t) values3d[0][0].size()};
  initMeta();
  for (uint32_t idx = 0; idx < shape_[0]; idx++) {
    for (uint32_t k = 0; k < shape_[1]; k++) {
      data_.insert(data_.end(), values3d[idx][k].begin(), values3d[idx][k].end());
    }
  }
}

void Tensor::initMeta() {
  dimCount_ = (uint32_t) shape_.size();
  elemCount_ = 1;
  strides_.resize(dimCount_);
  for (auto dim = int32_t(dimCount_ - 1); dim >= 0; dim--) {
    strides_[dim] = (int32_t) elemCount_;
    elemCount_ *= shape_[dim];
  }
}

void Tensor::initData() {
  data_.resize(elemCount_);
}

Tensor Tensor::reshape(const Shape &shape) {
  shape_ = shape;
  initMeta();
  initData();
  return *this;
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

Tensor Tensor::operator[](const std::vector<int32_t> &idx) const {
  std::vector<int32_t> realIdx;
  realIdx.reserve(idx.size());
  for (auto &i : idx) {
    realIdx.emplace_back(i >= 0 ? i : (i + shape()[0]));
  }

  Shape retShape = shape();
  retShape[0] = (uint32_t) realIdx.size();
  uint32_t dimStride = strides()[0];
  uint32_t retIdx = 0;
  Tensor retTensor = Tensor::shape(retShape);
  for (auto &i : realIdx) {
    memcpy(&retTensor[retIdx], &data()[i * dimStride], dimStride * sizeof(float));
    retIdx += dimStride;
  }

  return retTensor;
}

Tensor Tensor::transpose(const std::vector<uint32_t> &axis) const {
  TENSOR_CHECK_EMPTY(*this, {})
  if (dim() <= 1) {
    return *this;
  }

  TensorIter it(shape());
  if (axis.empty()) {
    // If not specified, defaults to range(a.ndim)[::-1], which reverses the order of the axes.
    std::vector<uint32_t> reverseTrans;
    reverseTrans.resize(dim());
    for (uint32_t i = 0; i < dim(); i++) {
      reverseTrans[i] = dim() - i - 1;
    }
    it.transpose(reverseTrans);
  } else if (axis.size() != dim()) {
    Tensor::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  } else {
    it.transpose(axis);
  }

  Tensor ret = Tensor::shape(it.shape());
  for (uint32_t idx = 0; idx < ret.elemCount_; idx++) {
    ret[idx] = data_[it.next()];
  }
  return ret;
}

std::vector<Tensor> Tensor::split(uint32_t sections, const Axis &axis) const {
  uint32_t axisDim = axis.get(dim());
  if (axisDim >= dim()) {
    Tensor::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  uint32_t dimSize = shape()[axisDim];
  if (dimSize % sections != 0) {
    Tensor::error(__FUNCTION__, TensorError_InvalidSections);
    return {};
  }

  // index of result tensors
  std::vector<uint32_t> splitIndices;
  splitIndices.resize(dimSize);

  uint32_t splitStride = dimSize / sections;
  uint32_t splitIdx = 0;
  uint32_t idx = 0;
  for (uint32_t i = 0; i < dimSize; i++) {
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
  for (uint32_t i = 0; i < sections; i++) {
    retTensors[i] = Tensor::shape(retShape);
  }

  // do split
  splitAxis(retTensors, splitIndices, axisDim);
  return retTensors;
}

std::vector<Tensor> Tensor::split(const std::vector<uint32_t> &indices, const Axis &axis) const {
  if (indices.empty()) {
    Tensor::error(__FUNCTION__, TensorError_InvalidSections);
    return {};
  }

  uint32_t axisDim = axis.get(dim());
  if (axisDim >= dim()) {
    Tensor::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // index of result tensors
  std::vector<uint32_t> splitIndices;
  uint32_t dimSize = shape()[axisDim];
  splitIndices.resize(dimSize);

  uint32_t splitIdx = 0;
  uint32_t idx = 0;
  for (uint32_t i = 0; i < dimSize; i++) {
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
  retShape[axisDim] = indices[0];
  retTensors[0] = Tensor::shape(retShape);
  // middle sections
  for (uint32_t i = 1; i < indices.size(); i++) {
    retShape[axisDim] = indices[i] - indices[i - 1];
    retTensors[i] = Tensor::shape(retShape);
  }
  // last section
  retShape[axisDim] = dimSize - indices.back();
  retTensors[indices.size()] = Tensor::shape(retShape);

  // do split
  splitAxis(retTensors, splitIndices, axisDim);
  return retTensors;
}

Tensor Tensor::concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  uint32_t totalSize = 0;
  for (auto &t : arrays) {
    totalSize += t.get().size();
  }

  Tensor retTensor = Tensor::shape({totalSize});
  uint32_t idx = 0;
  for (auto &t : arrays) {
    memcpy(&retTensor[idx], &t.get()[0], t.get().size() * sizeof(float));
    idx += t.get().size();
  }
  return retTensor;
}

Tensor Tensor::concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays, const Axis &axis) {
  // check axis
  auto &t0 = arrays[0].get();
  uint32_t axisDim = axis.get(t0.dim());
  if (axisDim >= t0.dim()) {
    Tensor::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // check shapes
  if (!checkShapeEqual(arrays, axisDim)) {
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // init concat tensor idx
  std::vector<uint32_t> concatIndices;
  for (uint32_t i = 0; i < arrays.size(); i++) {
    uint32_t dim = arrays[i].get().shape()[axisDim];
    for (uint32_t j = 0; j < dim; j++) {
      concatIndices.emplace_back(i);
    }
  }

  Shape retShape = t0.shape();
  retShape[axisDim] = (uint32_t) concatIndices.size();

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor Tensor::stack(const std::vector<std::reference_wrapper<Tensor>> &arrays, const Axis &axis) {
  // check axis
  auto &t0 = arrays[0].get();
  uint32_t axisDim = axis.get(t0.dim() + 1);
  if (axisDim > t0.dim()) {
    Tensor::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // check shapes
  for (uint32_t i = 1; i < arrays.size(); i++) {
    auto &t = arrays[i].get();
    if (t.shape() != t0.shape()) {
      Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
  }

  // init result shape
  Shape retShape = t0.shape();
  retShape.insert(retShape.begin() + axisDim, (uint32_t) arrays.size());
  Tensor retTensor = Tensor::shape(retShape);

  // do stack
  std::vector<uint32_t> srcIdx;
  srcIdx.resize(arrays.size());
  std::fill(srcIdx.begin(), srcIdx.end(), 0);

  int32_t dstIdx = 0;
  TensorIter it(retShape);
  while ((dstIdx = it.next()) >= 0) {
    uint32_t sectionId = it.coordinates()[axisDim];
    retTensor[dstIdx] = arrays[sectionId].get()[srcIdx[sectionId]++];
  }

  return retTensor;
}

Tensor Tensor::vstack(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  auto &t0 = arrays[0].get();
  uint32_t axisDim = 0;

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() == 1) {  // 1-D arrays must have the same length
    for (uint32_t i = 1; i < arrays.size(); i++) {
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
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // init concat tensor idx
  std::vector<uint32_t> concatIndices;
  if (t0.dim() == 1) {
    for (uint32_t i = 0; i < arrays.size(); i++) {
      concatIndices.emplace_back(i);
    }
  } else {
    for (uint32_t i = 0; i < arrays.size(); i++) {
      uint32_t dim = arrays[i].get().shape()[axisDim];
      for (uint32_t j = 0; j < dim; j++) {
        concatIndices.emplace_back(i);
      }
    }
  }

  Shape retShape;
  if (t0.dim() == 1) {
    retShape = {(uint32_t) concatIndices.size(), t0.shape()[0]};
  } else {
    retShape = t0.shape();
    retShape[axisDim] = (uint32_t) concatIndices.size();
  }

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor Tensor::hstack(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  auto &t0 = arrays[0].get();
  uint32_t axisDim = 1;

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() == 1) {  // 1-D arrays which can be any length
    for (uint32_t i = 1; i < arrays.size(); i++) {
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
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // This is equivalent to concatenation along the second axis,
  // except for 1-D arrays where it concatenates along the first axis
  if (t0.dim() == 1) {
    return concatenate(arrays);
  }

  // init concat tensor idx
  std::vector<uint32_t> concatIndices;
  for (uint32_t i = 0; i < arrays.size(); i++) {
    uint32_t dim = arrays[i].get().shape()[axisDim];
    for (uint32_t j = 0; j < dim; j++) {
      concatIndices.emplace_back(i);
    }
  }

  Shape retShape = t0.shape();
  retShape[axisDim] = (uint32_t) concatIndices.size();

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor Tensor::dstack(const std::vector<std::reference_wrapper<Tensor>> &arrays) {
  auto &t0 = arrays[0].get();
  uint32_t axisDim = 2;

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() <= 2) {  // 1-D or 2-D arrays must have the same shape
    for (uint32_t i = 1; i < arrays.size(); i++) {
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
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // init concat tensor idx
  std::vector<uint32_t> concatIndices;
  if (t0.dim() <= 2) {
    for (uint32_t i = 0; i < arrays.size(); i++) {
      concatIndices.emplace_back(i);
    }
  } else {
    for (uint32_t i = 0; i < arrays.size(); i++) {
      uint32_t dim = arrays[i].get().shape()[axisDim];
      for (uint32_t j = 0; j < dim; j++) {
        concatIndices.emplace_back(i);
      }
    }
  }

  Shape retShape;
  if (t0.dim() == 1) {
    retShape = {1, t0.shape()[0], (uint32_t) concatIndices.size()};
  } else if (t0.dim() == 2) {
    retShape = {t0.shape()[0], t0.shape()[1], (uint32_t) concatIndices.size()};
  } else {
    retShape = t0.shape();
    retShape[axisDim] = (uint32_t) concatIndices.size();
  }

  return arraysConcat(arrays, retShape, concatIndices, axisDim);
}

Tensor Tensor::operator+(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {})
  TENSOR_MATH_BROADCAST_PAIR(+)
}

Tensor Tensor::operator-(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {})
  TENSOR_MATH_BROADCAST_PAIR(-)
}

Tensor Tensor::operator*(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {})
  TENSOR_MATH_BROADCAST_PAIR(*)
}

Tensor Tensor::operator/(const Tensor &other) const {
  TENSOR_CHECK_EMPTY_PAIR(*this, other, {})
  TENSOR_MATH_BROADCAST_PAIR(/)
}

void Tensor::operator+=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other,)
  TENSOR_CHECK_SHAPE_EQUAL
  TENSOR_MATH_FAST_LOOP_SELF(+=, other[idx])
}

void Tensor::operator-=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other,)
  TENSOR_CHECK_SHAPE_EQUAL
  TENSOR_MATH_FAST_LOOP_SELF(-=, other[idx])
}

void Tensor::operator*=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other,)
  TENSOR_CHECK_SHAPE_EQUAL
  TENSOR_MATH_FAST_LOOP_SELF(*=, other[idx])
}

void Tensor::operator/=(const Tensor &other) {
  TENSOR_CHECK_EMPTY_PAIR(*this, other,)
  TENSOR_CHECK_SHAPE_EQUAL
  TENSOR_MATH_FAST_LOOP_SELF(/=, other[idx])
}

Tensor Tensor::operator+(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {})
  TENSOR_MATH_FAST_LOOP_PAIR(*this, +=, other)
}

Tensor Tensor::operator-(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {})
  TENSOR_MATH_FAST_LOOP_PAIR(*this, -=, other)
}

Tensor Tensor::operator*(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {})
  TENSOR_MATH_FAST_LOOP_PAIR(*this, *=, other)
}

Tensor Tensor::operator/(const float &other) const {
  TENSOR_CHECK_EMPTY(*this, {})
  TENSOR_MATH_FAST_LOOP_PAIR(*this, /=, other)
}

void Tensor::operator+=(const float &other) {
  TENSOR_CHECK_EMPTY(*this,)
  TENSOR_MATH_FAST_LOOP_SELF(+=, other)
}

void Tensor::operator-=(const float &other) {
  TENSOR_CHECK_EMPTY(*this,)
  TENSOR_MATH_FAST_LOOP_SELF(-=, other)
}

void Tensor::operator*=(const float &other) {
  TENSOR_CHECK_EMPTY(*this,)
  TENSOR_MATH_FAST_LOOP_SELF(*=, other)
}

void Tensor::operator/=(const float &other) {
  TENSOR_CHECK_EMPTY(*this,)
  TENSOR_MATH_FAST_LOOP_SELF(/=, other)
}

Tensor operator+(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {})
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other + ret[idx])
}

Tensor operator-(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {})
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other - ret[idx])
}

Tensor operator*(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {})
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other * ret[idx])
}

Tensor operator/(const float &other, const Tensor &obj) {
  TENSOR_CHECK_EMPTY(obj, {})
  TENSOR_MATH_FAST_LOOP_PAIR(obj, =, other / ret[idx])
}

Tensor Tensor::sqrt(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, std::sqrt(ret[idx]))
}

Tensor Tensor::tanh(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, fastTanh(ret[idx]))
}

Tensor Tensor::exp(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_MATH_FAST_LOOP_PAIR(t, =, std::exp(ret[idx]))
}

float Tensor::dot(const float &a, const float &b) {
  return a * b;
}

Tensor Tensor::dot(const Tensor &a, const float &b) {
  TENSOR_CHECK_EMPTY(a, {})
  return a * b;
}

Tensor Tensor::dot(const float &a, const Tensor &b) {
  TENSOR_CHECK_EMPTY(b, {})
  return b * a;
}

Tensor Tensor::dot(const Tensor &a, const Tensor &b) {
  TENSOR_CHECK_EMPTY_PAIR(a, b, {})

  // If both a and b are 1-D arrays, it is inner product of vectors
  if (a.dim() == 1 && b.dim() == 1) {
    if (a.size() != b.size()) {
      Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
    float ret = 0.f;
    for (uint32_t i = 0; i < a.size(); i++) {
      ret += a[i] * b[i];
    }
    return Tensor::scalar(ret);
  }

  // If both a and b are 2-D arrays, it is matrix multiplication
  if (a.dim() == 2 && b.dim() == 2) {
    uint32_t m = a.shape()[0];
    uint32_t middle = a.shape()[1];
    if (middle != b.shape()[0]) {
      Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
    uint32_t n = b.shape()[1];
    Tensor ret = Tensor::shape({m, n});
    Blas::gemm(&ret[0], &a[0], &b[0], m, middle, n);
    return ret;
  }

  // If either a or b is 0-D (scalar), it is equivalent to multiply
  if (a.isScalar()) { return b * a[0]; }
  if (b.isScalar()) { return a * b[0]; }

  // If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
  // If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last
  // axis of a and the second-to-last axis of b:
  Tensor::error(__FUNCTION__, TensorError_NotSupport);
  return {};
}

Tensor Tensor::matmul(const Tensor &a, const Tensor &b) {
  TENSOR_CHECK_EMPTY_PAIR(a, b, {})

  // Multiplication by scalars is not allowed, use * instead.
  if (a.isScalar() || b.isScalar()) {
    Tensor::error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }

  // rules:
  // If both arguments are 2-D they are multiplied like conventional matrices.
  // If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
  // If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

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
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // check shape broadcast compatible
  Shape retShape;
  auto compatible = checkCompatible(shapeA, shapeB, retShape, 2);
  if (compatible == ShapeCompatible_Error) {
    Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  auto retDimCnt = (uint32_t) retShape.size();
  auto m = shapeA[shapeA.size() - 2];
  auto k = shapeA.back();
  auto n = shapeB.back();

  retShape[retDimCnt - 2] = m;
  retShape[retDimCnt - 1] = n;
  Tensor retTensor = Tensor::shape(retShape);
  if (retDimCnt > 2) {
    // broadcast matrix multiplication
    auto broadcastShape = Shape(retShape.begin(), retShape.end() - 2);
    TensorIter iterRet(broadcastShape);
    TensorIter iterA = shapeA.size() > 2 ? TensorIter({shapeA.begin(), shapeA.end() - 2}) : TensorIter({1});
    TensorIter iterB = shapeB.size() > 2 ? TensorIter({shapeB.begin(), shapeB.end() - 2}) : TensorIter({1});
    iterA.broadcast(broadcastShape);
    iterB.broadcast(broadcastShape);
    auto strideRet = m * n;
    auto strideA = m * k;
    auto strideB = k * n;
    for (uint32_t idx = 0; idx < iterRet.size(); idx++) {
      Blas::gemm(&retTensor[iterRet.next() * strideRet], &a[iterA.next() * strideA], &b[iterB.next() * strideB], m, k, n);
    }
  } else {
    Blas::gemm(&retTensor[0], &a[0], &b[0], m, k, n);
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
    uint32_t m = a.shape()[0];
    uint32_t k = a.shape()[1];
    uint32_t n = b.shape()[0];
    if (k != b.shape()[1]) {
      Tensor::error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
    Tensor retTensor = Tensor::shape({m, n});
    Blas::gemmTrans(&retTensor[0], &a[0], &b[0], m, k, n);
    return retTensor;
  }

  // slow path
  return matmul(a, b.transpose());
}

float Tensor::min(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0)
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncMin)
}

float Tensor::max(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0)
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncMax)
}

float Tensor::mean(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0)
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncMean)
}

float Tensor::sum(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0)
  TENSOR_UFUNC_FAST_LOOP(t[0], UFuncSum)
}

float Tensor::var(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0)
  TENSOR_UFUNC_FAST_LOOP(0, UFuncVar)
}

float Tensor::argmin(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0)
  TENSOR_UFUNC_FAST_LOOP(0, UFuncArgMin)
}

float Tensor::argmax(const Tensor &t) {
  TENSOR_CHECK_EMPTY(t, 0)
  TENSOR_UFUNC_FAST_LOOP(0, UFuncArgMax)
}

Tensor Tensor::min(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_UFUNC_REDUCE(UFuncMin)
}

Tensor Tensor::max(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_UFUNC_REDUCE(UFuncMax)
}

Tensor Tensor::mean(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_UFUNC_REDUCE(UFuncMean)
}

Tensor Tensor::sum(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_UFUNC_REDUCE(UFuncSum)
}

Tensor Tensor::var(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_UFUNC_REDUCE(UFuncVar)
}

Tensor Tensor::argmin(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_UFUNC_REDUCE(UFuncArgMin)
}

Tensor Tensor::argmax(const Tensor &t, const Axis &axis, bool keepDims) {
  TENSOR_CHECK_EMPTY(t, {})
  TENSOR_UFUNC_REDUCE(UFuncArgMax)
}

void Tensor::traverse(UFunc &func, uint32_t start, uint32_t stride, uint32_t cnt) const {
  uint32_t idx = start;
  for (uint32_t n = 0; n < cnt; n++) {
    func.op(data_[idx]);
    idx += stride;
  }
}

Tensor Tensor::reduce(UFunc &func, uint32_t axis, bool keepDims) const {
  // check axis
  if (axis >= dimCount_) {
    Tensor::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // construct result shape
  Shape retShape;
  retShape.reserve(dimCount_);
  for (uint32_t dim = 0; dim < dimCount_; dim++) {
    if (axis == dim) {
      if (keepDims) {
        retShape.emplace_back(1);
      }
    } else {
      retShape.emplace_back(shape_[dim]);
    }
  }

  // reduce via function
  Tensor ret = Tensor::shape(retShape);

  uint32_t axisStride = strides_[axis];
  uint32_t axisLength = shape_[axis];

  uint32_t groupStride = axisStride * axisLength;
  uint32_t groupCount = elemCount_ / groupStride;

  uint32_t retIdx = 0;
  uint32_t axisStart = 0;
  for (uint32_t i = 0; i < groupCount; i++) {
    axisStart = i * groupStride;
    for (uint32_t j = 0; j < axisStride; j++) {
      func.reset();
      traverse(func, axisStart, axisStride, axisLength);
      ret[retIdx++] = func.result();
      axisStart++;
    }
  }

  return ret;
}

void Tensor::splitAxis(std::vector<Tensor> &retTensors, std::vector<uint32_t> &splitIndices, uint32_t axis) const {
  std::vector<uint32_t> dstIdx;
  dstIdx.resize(retTensors.size());
  std::fill(dstIdx.begin(), dstIdx.end(), 0);

  TensorIter it(shape());
  int32_t srcIdx = 0;
  while ((srcIdx = it.next()) >= 0) {
    uint32_t sectionId = splitIndices[it.coordinates()[axis]];
    retTensors[sectionId][dstIdx[sectionId]++] = data_[srcIdx];
  }
}

Tensor Tensor::arraysConcat(const std::vector<std::reference_wrapper<Tensor>> &arrays, const Shape &retShape,
                            const std::vector<uint32_t> &concatIndices, uint32_t axis) {
  Tensor retTensor = Tensor::shape(retShape);

  // do concat
  std::vector<uint32_t> srcIdx;
  srcIdx.resize(arrays.size());
  std::fill(srcIdx.begin(), srcIdx.end(), 0);

  int32_t dstIdx = 0;
  TensorIter it(retShape);
  while ((dstIdx = it.next()) >= 0) {
    uint32_t sectionId = concatIndices[it.coordinates()[axis]];
    retTensor[dstIdx] = arrays[sectionId].get()[srcIdx[sectionId]++];
  }

  return retTensor;
}

ShapeCompatible Tensor::checkCompatible(const Shape &t0, const Shape &t1, Shape &retShape, uint32_t skipLast) {
  retShape = t0.size() > t1.size() ? t0 : t1;

  auto idxRet = (int32_t) (retShape.size() - 1 - skipLast);
  auto idx0 = (int32_t) (t0.size() - 1 - skipLast);
  auto idx1 = (int32_t) (t1.size() - 1 - skipLast);

  bool needBroadcast = false;  // dimensions already exist
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

bool Tensor::checkShapeEqual(const std::vector<std::reference_wrapper<Tensor>> &arrays, uint32_t exceptAxis) {
  auto &t0 = arrays[0].get();

  for (uint32_t i = 1; i < arrays.size(); i++) {
    auto &t = arrays[i].get();
    if (t.dim() != t0.dim()) {
      return false;
    }
    for (uint32_t j = 0; j < t.dim(); j++) {
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

TensorIter::TensorIter(const Shape &shape) {
  reshape(shape);
}

Shape TensorIter::shape() {
  Shape ret;
  ret.resize(ndM1_ + 1);
  for (int32_t i = 0; i <= ndM1_; i++) {
    ret[i] = dimsM1_[i] + 1;
  }
  return ret;
}

void TensorIter::reshape(const Shape &shape) {
  ndM1_ = (int32_t) shape.size() - 1;
  size_ = 1;
  for (auto dim = int32_t(ndM1_); dim >= 0; dim--) {
    dimsM1_[dim] = (int32_t) shape[dim] - 1;
    strides_[dim] = (int32_t) size_;
    backStrides_[dim] = strides_[dim] * dimsM1_[dim];

    size_ *= (int32_t) shape[dim];
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
    for (auto dim = (int32_t) ndM1_; dim >= 0; dim--) {
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
  uint32_t targetNdM1_ = (uint32_t) shape.size() - 1;

  // origin dimensions
  for (auto dim = (int32_t) ndM1_; dim >= 0; dim--) {
    uint32_t targetDim = targetNdM1_ - ndM1_ + dim;
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
  for (uint32_t dim_ = 0; dim_ < targetNdM1_ - ndM1_; dim_++) {
    strides_[dim_] = 0;
    backStrides_[dim_] = 0;
  }

  // update shape
  ndM1_ = (int32_t) shape.size() - 1;
  size_ = 1;
  for (auto dim = int32_t(ndM1_); dim >= 0; dim--) {
    dimsM1_[dim] = (int32_t) shape[dim] - 1;
    size_ *= (int32_t) shape[dim];
  }

  // reset
  reset();
}

void TensorIter::transpose(const std::vector<uint32_t> &axis) {
  // assume axis size equal to dimension count
  assert(axis.size() == ndM1_ + 1);

  // reorder dimsM1_, strides_, backStrides_
  reorder(dimsM1_, axis);
  reorder(strides_, axis);
  reorder(backStrides_, axis);
}

}
