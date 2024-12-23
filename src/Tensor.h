/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace TinyGPT {

#define TENSOR_MAX_DIMS 16

typedef enum TensorError_ {
  TensorError_None = 0,
  TensorError_EmptyTensor,
  TensorError_InvalidShape,
  TensorError_InvalidAxis,
  TensorError_InvalidSections,
  TensorError_ShapeNotAligned,
  TensorError_NotSupport,
} TensorError;

typedef enum ShapeCompatible_ {
  ShapeCompatible_Error = 0,
  ShapeCompatible_SameShape,
  ShapeCompatible_Broadcast,
} ShapeCompatible;

typedef std::vector<int32_t> Shape;
typedef std::vector<float> Array1d;
typedef std::vector<std::vector<float>> Array2d;
typedef std::vector<std::vector<std::vector<float>>> Array3d;

class RandomGenerator {
public:
  static void setSeed(const unsigned int seed) {
    seed_ = seed;
    randomEngine_ = std::default_random_engine(seed_.value());
  }
  static std::default_random_engine getGenerator() {
    if (seed_.has_value()) {
      return randomEngine_;
    }
    std::random_device r;
    return std::default_random_engine(r());
  }

private:
  static std::optional<unsigned int> seed_;
  static std::default_random_engine randomEngine_;
};

struct Size2D {
  Size2D(int32_t n) : h(n), w(n) {}
  Size2D(int32_t h, int32_t w) : h(h), w(w) {}

  int32_t h = 0;
  int32_t w = 0;
};

// one axis only
class Axis {
public:
  Axis() = delete;

  Axis(int32_t axis) : axis_(axis) {}

  [[nodiscard]] int32_t get(int32_t axisCnt) const {
    return axis_ >= 0 ? axis_ : axis_ + axisCnt;
  }

private:
  int32_t axis_ = 0;
};

class UFunc {
public:
  virtual ~UFunc() = default;
  virtual void op(const float &val) { idx_++; };

  virtual float result() { return tmp; };

  virtual void reset() {
    idx_ = 0;
    tmp = 0.f;
  }

protected:
  int32_t idx_ = 0;
  float tmp = 0.f;
};

class UFuncSum : public UFunc {
public:
  void op(const float &val) override { tmp += val; }
};

class UFuncMean : public UFunc {
public:
  void op(const float &val) override {
    idx_++;
    tmp += val;
  }

  float result() override { return tmp / (float)idx_; }
};

class UFuncVar : public UFunc {
public:
  void op(const float &val) override {
    idx_++;
    tmp += val;
    squareSum_ += val * val;
  }

  float result() override {
    float mean = tmp / (float)idx_;
    return squareSum_ / (float)idx_ - mean * mean;
  }

  void reset() override {
    idx_ = 0;
    tmp = 0;
    squareSum_ = 0;
  }

private:
  float squareSum_ = 0;
};

class UFuncMin : public UFunc {
public:
  void op(const float &val) override {
    if (val < tmp) {
      tmp = val;
    }
  }

  void reset() override { tmp = std::numeric_limits<float>::max(); }
};

class UFuncMax : public UFunc {
public:
  void op(const float &val) override {
    if (val > tmp) {
      tmp = val;
    }
  }

  void reset() override { tmp = -std::numeric_limits<float>::max(); }
};

class UFuncArgMin : public UFunc {
public:
  void op(const float &val) override {
    if (val < tmp) {
      tmp = val;
      minIdx_ = idx_;
    }
    idx_++;
  }

  float result() override { return (float)minIdx_; }

  void reset() override {
    tmp = std::numeric_limits<float>::max();
    idx_ = 0;
    minIdx_ = 0;
  }

private:
  int32_t minIdx_ = 0;
};

class UFuncArgMax : public UFunc {
public:
  void op(const float &val) override {
    if (val > tmp) {
      tmp = val;
      maxIdx_ = idx_;
    }
    idx_++;
  }

  float result() override { return (float)maxIdx_; }

  void reset() override {
    tmp = -std::numeric_limits<float>::max();
    idx_ = 0;
    maxIdx_ = 0;
  }

private:
  int32_t maxIdx_ = 0;
};

// float type elements only
class Tensor {
public:
  Tensor() = default;

  Tensor(const Tensor &other) {
    dispose();
    copyFrom(other);
    initData(other.data_);
  }

  Tensor(Tensor &&other) noexcept {
    copyFrom(other);
    other.data_ = nullptr;
  }

  Tensor &operator=(const Tensor &other) {
    if (this == &other) {
      return *this;
    }
    dispose();
    copyFrom(other);
    initData(other.data_);
    return *this;
  }

  Tensor &operator=(Tensor &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    dispose();
    copyFrom(other);
    other.data_ = nullptr;
    return *this;
  }

  void copyFrom(const Tensor &other) {
    dimCount_ = other.dimCount_;
    elemCount_ = other.elemCount_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    data_ = other.data_;
  }

  void dispose() {
    dimCount_ = 0;
    elemCount_ = 0;
    shape_.clear();
    strides_.clear();
    delete[] data_;
    data_ = nullptr;
  }

  ~Tensor() { dispose(); }

  static Tensor shape(const Shape &shape);

  static Tensor scalar(const float &value);

  static Tensor ones(const Shape &shape);

  static Tensor onesLike(const Tensor &t);

  static Tensor zeros(const Shape &shape);

  static Tensor rand(const Shape &shape);

  static Tensor randn(const Shape &shape);

  static Tensor bernoulli(const Shape &shape, float p);

  static Tensor tri(int32_t n, int32_t m = 0, int32_t k = 0);

  // 1d array
  explicit Tensor(const Array1d &values1d);
  // 2d array
  explicit Tensor(const Array2d &values2d);
  // 3d array
  explicit Tensor(const Array3d &values3d);

  Tensor reshape(const Shape &shape);
  static Tensor reshape(const Tensor &t, const Shape &shape);
  [[nodiscard]] Tensor reshape(const Shape &shape) const;
  [[nodiscard]] Tensor view(const Shape &shape) const { return reshape(shape); }

  void flatten(int32_t startDim = 0, int32_t endDim = -1);
  static Tensor flatten(const Tensor &t, int32_t startDim = 0,
                        int32_t endDim = -1);
  void unflatten(int32_t dim, const std::vector<int32_t> &sizes);
  static Tensor unflatten(const Tensor &t, int32_t dim,
                          const std::vector<int32_t> &sizes);

  void squeeze(int32_t dim = -1);
  void squeeze(const std::vector<int32_t> &dims);
  static Tensor squeeze(const Tensor &t, int32_t dim = -1);
  static Tensor squeeze(const Tensor &t, const std::vector<int32_t> &dims);
  void unsqueeze(int32_t dim);
  static Tensor unsqueeze(const Tensor &t, int32_t dim);

  [[nodiscard]] bool empty() const { return elemCount_ == 0; }

  [[nodiscard]] bool isScalar() const {
    return dimCount_ == 0 && elemCount_ == 1;
  }

  [[nodiscard]] int32_t dim() const { return dimCount_; }

  [[nodiscard]] int32_t size() const { return elemCount_; }

  [[nodiscard]] const Shape &shape() const { return shape_; }

  [[nodiscard]] const std::vector<int32_t> &strides() const { return strides_; }

  [[nodiscard]] float item() const { return data_[0]; }

  float &operator[](int32_t idx) { return data_[idx]; }

  const float &operator[](int32_t idx) const { return data_[idx]; }

  template <typename T = float> [[nodiscard]] std::vector<T> toArray() const;

  // fill
  void fill(float value);
  static Tensor fill(const Tensor &t, float value);

  // clamp
  void clampMin(float min);
  void clampMax(float max);
  void clamp(float min, float max);

  static Tensor clampMin(const Tensor &t, float min);
  static Tensor clampMax(const Tensor &t, float max);
  static Tensor clamp(const Tensor &t, float min, float max);

  // range
  static std::vector<int32_t> range(int32_t start, int32_t stop,
                                    int32_t step = 1);
  static Tensor arange(float start, float stop, float step = 1.f);
  static Tensor linspace(float start, float end, int steps);

  // indexing
  template <typename... Args> [[nodiscard]] Tensor index(Args... args) const {
    std::vector<int32_t> vec;
    vec.reserve(sizeof...(args));
    (vec.push_back(args), ...);
    return indexInteger(vec);
  }
  Tensor indexInteger(const std::vector<int32_t> &idx,
                      float *dataPtr = nullptr) const;
  [[nodiscard]] Tensor index(const std::vector<int32_t> &idx) const;
  [[nodiscard]] Tensor
  indexAdvance(const std::vector<std::vector<int32_t>> &indexes) const;

  void indexIntegerSet(const std::vector<int32_t> &idx, float val);
  void indexIntegerSet(const std::vector<int32_t> &idx, const Tensor &val);
  void indexAdvanceSet(const std::vector<std::vector<int32_t>> &indexes,
                       float val);
  void indexAdvanceSet(const std::vector<std::vector<int32_t>> &indexes,
                       const Tensor &val);

  // im2col
  [[nodiscard]] Tensor im2col(Size2D kernelSize, Size2D stride,
                              Size2D padding = 0) const;
  // col2im
  [[nodiscard]] Tensor col2im(const Shape &inputShape, Size2D kernelSize,
                              Size2D stride, Size2D padding = 0) const;

  // transpose
  [[nodiscard]] Tensor transpose(const std::vector<int32_t> &axis = {}) const;

  static Tensor transpose(const Tensor &t,
                          const std::vector<int32_t> &axis = {}) {
    return t.transpose(axis);
  }

  // split
  [[nodiscard]] std::vector<Tensor> split(int32_t sections,
                                          const Axis &axis = 0) const;

  [[nodiscard]] std::vector<Tensor> vsplit(int32_t sections) const {
    return split(sections, 0);
  }

  [[nodiscard]] std::vector<Tensor> hsplit(int32_t sections) const {
    return split(sections, 1);
  }

  [[nodiscard]] std::vector<Tensor> dsplit(int32_t sections) const {
    return split(sections, 2);
  }

  [[nodiscard]] std::vector<Tensor> split(const std::vector<int32_t> &indices,
                                          const Axis &axis = 0) const;

  [[nodiscard]] std::vector<Tensor>
  vsplit(const std::vector<int32_t> &indices) const {
    return split(indices, 0);
  }

  [[nodiscard]] std::vector<Tensor>
  hsplit(const std::vector<int32_t> &indices) const {
    return split(indices, 1);
  }

  [[nodiscard]] std::vector<Tensor>
  dsplit(const std::vector<int32_t> &indices) const {
    return split(indices, 2);
  }

  static std::vector<Tensor> split(const Tensor &t, int32_t sections,
                                   const Axis &axis = 0) {
    return t.split(sections, axis);
  }

  static std::vector<Tensor> vsplit(const Tensor &t, int32_t sections) {
    return t.split(sections, 0);
  }

  static std::vector<Tensor> hsplit(const Tensor &t, int32_t sections) {
    return t.split(sections, 1);
  }

  static std::vector<Tensor> dsplit(const Tensor &t, int32_t sections) {
    return t.split(sections, 2);
  }

  static std::vector<Tensor> split(const Tensor &t,
                                   const std::vector<int32_t> &indices,
                                   const Axis &axis = 0) {
    return t.split(indices, axis);
  }

  static std::vector<Tensor> vsplit(const Tensor &t,
                                    const std::vector<int32_t> &indices) {
    return t.split(indices, 0);
  }

  static std::vector<Tensor> hsplit(const Tensor &t,
                                    const std::vector<int32_t> &indices) {
    return t.split(indices, 1);
  }

  static std::vector<Tensor> dsplit(const Tensor &t,
                                    const std::vector<int32_t> &indices) {
    return t.split(indices, 2);
  }

  // concatenate
  static Tensor
  concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays);
  static Tensor
  concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays,
              const Axis &axis);

  // stack
  static Tensor stack(const std::vector<std::reference_wrapper<Tensor>> &arrays,
                      const Axis &axis = 0);
  static Tensor
  vstack(const std::vector<std::reference_wrapper<Tensor>> &arrays);
  static Tensor
  hstack(const std::vector<std::reference_wrapper<Tensor>> &arrays);
  static Tensor
  dstack(const std::vector<std::reference_wrapper<Tensor>> &arrays);

  // compare
  Tensor operator<(const Tensor &other) const;
  Tensor operator>(const Tensor &other) const;
  Tensor operator==(const Tensor &other) const;
  Tensor operator!=(const Tensor &other) const;
  Tensor operator<(const float &other) const;
  Tensor operator>(const float &other) const;
  Tensor operator==(const float &other) const;
  Tensor operator!=(const float &other) const;

  // math
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;

  Tensor operator+(const float &other) const;
  Tensor operator-(const float &other) const;
  Tensor operator*(const float &other) const;
  Tensor operator/(const float &other) const;

  void operator+=(const Tensor &other);
  void operator-=(const Tensor &other);
  void operator*=(const Tensor &other);
  void operator/=(const Tensor &other);

  void operator+=(const float &other);
  void operator-=(const float &other);
  void operator*=(const float &other);
  void operator/=(const float &other);

  friend Tensor operator+(const float &other, const Tensor &obj);
  friend Tensor operator-(const float &other, const Tensor &obj);
  friend Tensor operator*(const float &other, const Tensor &obj);
  friend Tensor operator/(const float &other, const Tensor &obj);

  static Tensor sin(const Tensor &t);
  static Tensor cos(const Tensor &t);
  static Tensor sqrt(const Tensor &t);
  static Tensor tanh(const Tensor &t);
  static Tensor exp(const Tensor &t);
  static Tensor log(const Tensor &t);

  [[nodiscard]] Tensor sin() const { return Tensor::sin(*this); }
  [[nodiscard]] Tensor cos() const { return Tensor::cos(*this); }
  [[nodiscard]] Tensor sqrt() const { return Tensor::sqrt(*this); }
  [[nodiscard]] Tensor tanh() const { return Tensor::tanh(*this); }
  [[nodiscard]] Tensor exp() const { return Tensor::exp(*this); }
  [[nodiscard]] Tensor log() const { return Tensor::log(*this); }

  [[nodiscard]] Tensor pow(const Tensor &other) const;
  [[nodiscard]] Tensor pow(const float &other) const;

  static Tensor pow(const Tensor &x1, const Tensor &x2) { return x1.pow(x2); }
  static Tensor pow(const Tensor &x1, const float &x2) { return x1.pow(x2); }

  // linear algebra
  static float dot(const float &a, const float &b);
  static Tensor dot(const Tensor &a, const float &b);
  static Tensor dot(const float &a, const Tensor &b);
  static Tensor dot(const Tensor &a, const Tensor &b);
  static Tensor matmul(const Tensor &a, const Tensor &b);
  static Tensor matmulTrans(const Tensor &a, const Tensor &b);

  // aggregation

  static float min(const Tensor &t);
  static float max(const Tensor &t);
  static float mean(const Tensor &t);
  static float sum(const Tensor &t);
  static float var(const Tensor &t);
  static float argmin(const Tensor &t);
  static float argmax(const Tensor &t);

  [[nodiscard]] float min() const { return Tensor::min(*this); };
  [[nodiscard]] float max() const { return Tensor::max(*this); };
  [[nodiscard]] float mean() const { return Tensor::mean(*this); };
  [[nodiscard]] float sum() const { return Tensor::sum(*this); };
  [[nodiscard]] float var() const { return Tensor::var(*this); };
  [[nodiscard]] float argmin() const { return Tensor::argmin(*this); };
  [[nodiscard]] float argmax() const { return Tensor::argmax(*this); };

  static Tensor min(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor max(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor mean(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor sum(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor var(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor argmin(const Tensor &t, const Axis &axis,
                       bool keepDims = false);
  static Tensor argmax(const Tensor &t, const Axis &axis,
                       bool keepDims = false);

  [[nodiscard]] Tensor min(const Axis &axis, bool keepDims = false) const {
    return Tensor::min(*this, axis, keepDims);
  }

  [[nodiscard]] Tensor max(const Axis &axis, bool keepDims = false) const {
    return Tensor::max(*this, axis, keepDims);
  }

  [[nodiscard]] Tensor mean(const Axis &axis, bool keepDims = false) const {
    return Tensor::mean(*this, axis, keepDims);
  }

  [[nodiscard]] Tensor sum(const Axis &axis, bool keepDims = false) const {
    return Tensor::sum(*this, axis, keepDims);
  }

  [[nodiscard]] Tensor var(const Axis &axis, bool keepDims = false) const {
    return Tensor::var(*this, axis, keepDims);
  }

  [[nodiscard]] Tensor argmin(const Axis &axis, bool keepDims = false) const {
    return Tensor::argmin(*this, axis, keepDims);
  }

  [[nodiscard]] Tensor argmax(const Axis &axis, bool keepDims = false) const {
    return Tensor::argmax(*this, axis, keepDims);
  }

public:
  class Iterator {
  public:
    explicit Iterator(const float *ptr) : ptr(ptr) {}
    const float &operator*() const { return *ptr; }

    Iterator &operator++() {
      ++ptr;
      return *this;
    }

    bool operator==(const Iterator &other) const { return ptr == other.ptr; }
    bool operator!=(const Iterator &other) const { return ptr != other.ptr; }

  private:
    const float *ptr;
  };

  [[nodiscard]] Iterator begin() const { return Iterator(data_); }
  [[nodiscard]] Iterator end() const { return Iterator(data_ + elemCount_); }

protected:
  void initMeta();
  void initData(const float *from = nullptr);

  void traverse(UFunc &func, int32_t start, int32_t stride, int32_t cnt) const;
  Tensor reduce(UFunc &func, int32_t axis, bool keepDims = false) const;
  void splitAxis(std::vector<Tensor> &retTensors,
                 std::vector<int32_t> &splitIndices, int32_t axis) const;

  static Tensor
  arraysConcat(const std::vector<std::reference_wrapper<Tensor>> &arrays,
               const Shape &retShape, const std::vector<int32_t> &concatIndices,
               int32_t axis);
  static ShapeCompatible checkCompatible(const Shape &t0, const Shape &t1,
                                         Shape &retShape, int32_t skipLast = 0);
  static bool
  checkShapeEqual(const std::vector<std::reference_wrapper<Tensor>> &arrays,
                  int32_t exceptAxis);
  static void error(const char *where, TensorError error);

private:
  static float fastTanh(float x);
  void indexIntegerSet(const std::vector<int32_t> &idx, const float *valPtr);

protected:
  int32_t dimCount_ = 0;
  int32_t elemCount_ = 0;
  Shape shape_;
  std::vector<int32_t> strides_;
  float *data_ = nullptr;
};

template <typename T> std::vector<T> Tensor::toArray() const {
  std::vector<T> ret;
  ret.reserve(elemCount_);
  for (int32_t i = 0; i < elemCount_; i++) {
    ret.push_back((T)data_[i]);
  }
  return ret;
}

class TensorIter {
public:
  explicit TensorIter(const Shape &shape);

  // get shape
  Shape shape();

  // reshape
  void reshape(const Shape &shape);

  // get size
  [[nodiscard]] int32_t size() const { return size_; }

  // get current coordinates
  [[nodiscard]] const int32_t *coordinates() const { return coordinates_; };

  // return -1 if not available
  int32_t next();

  // reset to init states
  void reset();

  // broadcast to shape (no broadcast rules check)
  void broadcast(const Shape &shape);

  // transpose
  void transpose(const std::vector<int32_t> &axis);

protected:
  // reorder array
  static void reorder(int32_t *v, const std::vector<int32_t> &order) {
    auto n = order.size();
    std::vector<int32_t> temp(n);
    for (int i = 0; i < n; ++i) {
      temp[i] = v[order[i]];
    }
    memcpy(v, temp.data(), sizeof(int32_t) * n);
  }

protected:
  int32_t ndM1_ = 0;
  int32_t size_ = 0;
  int32_t dimsM1_[TENSOR_MAX_DIMS]{};

  int32_t strides_[TENSOR_MAX_DIMS]{};
  int32_t backStrides_[TENSOR_MAX_DIMS]{};

  int32_t coordinates_[TENSOR_MAX_DIMS]{};
  int32_t index_ = 0;
  int32_t itCnt_ = 0;
};

} // namespace TinyGPT
