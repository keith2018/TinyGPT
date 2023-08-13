/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <limits>
#include <cmath>

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

typedef std::vector<uint32_t> Shape;
typedef std::vector<float> Array1d;
typedef std::vector<std::vector<float>> Array2d;
typedef std::vector<std::vector<std::vector<float>>> Array3d;

// one axis only
class Axis {
 public:
  Axis() = delete;
  Axis(int32_t axis) : axis_(axis) {}

  uint32_t get(uint32_t axisCnt) const {
    return axis_ >= 0 ? axis_ : axis_ + axisCnt;
  }

 private:
  int32_t axis_ = 0;
};

class UFunc {
 public:
  virtual void op(const float &val) {
    idx_++;
  };

  virtual float result() {
    return tmp;
  };

  virtual void reset() {
    idx_ = 0;
    tmp = 0.f;
  }

 protected:
  uint32_t idx_ = 0;
  float tmp = 0.f;
};

class UFuncSum : public UFunc {
 public:
  void op(const float &val) override {
    tmp += val;
  }
};

class UFuncMean : public UFunc {
 public:
  void op(const float &val) override {
    idx_++;
    tmp += val;
  }

  float result() override {
    return tmp / (float) idx_;
  }
};

class UFuncVar : public UFunc {
 public:
  void op(const float &val) override {
    idx_++;
    tmp += val;
    squareSum_ += val * val;
  }

  float result() override {
    float mean = tmp / (float) idx_;
    return squareSum_ / (float) idx_ - mean * mean;
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

  void reset() override {
    tmp = std::numeric_limits<float>::max();
  }
};

class UFuncMax : public UFunc {
 public:
  void op(const float &val) override {
    if (val > tmp) {
      tmp = val;
    }
  }

  void reset() override {
    tmp = -std::numeric_limits<float>::max();
  }
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

  float result() override {
    return (float) minIdx_;
  }

  void reset() override {
    tmp = std::numeric_limits<float>::max();
    idx_ = 0;
    minIdx_ = 0;
  }

 private:
  uint32_t minIdx_ = 0;
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

  float result() override {
    return (float) maxIdx_;
  }

  void reset() override {
    tmp = -std::numeric_limits<float>::max();
    idx_ = 0;
    maxIdx_ = 0;
  }

 private:
  uint32_t maxIdx_ = 0;
};

// float type elements only
class Tensor {
 public:
  // constructor

  Tensor() = default;

  static Tensor shape(const Shape &shape);

  static Tensor scalar(const float &value);

  static Tensor ones(const Shape &shape);

  static Tensor zeros(const Shape &shape);

  static Tensor tri(uint32_t n, uint32_t m = 0, int32_t k = 0);

  // 1d array
  explicit Tensor(const Array1d &values1d);
  // 2d array
  explicit Tensor(const Array2d &values2d);
  // 3d array
  explicit Tensor(const Array3d &values3d);

  Tensor reshape(const Shape &shape);

  inline bool empty() const {
    return data_.empty();
  }

  inline bool isScalar() const {
    return dimCount_ == 0 && elemCount_ == 1;
  }

  inline uint32_t dim() const {
    return dimCount_;
  }

  inline uint32_t size() const {
    return elemCount_;
  }

  inline const Shape &shape() const {
    return shape_;
  }

  inline const std::vector<int32_t> &strides() const {
    return strides_;
  }

  inline std::vector<float> &data() {
    return data_;
  }

  inline const std::vector<float> &data() const {
    return data_;
  }

  // range
  static std::vector<int32_t> range(int32_t start, int32_t stop, int32_t step = 1);
  static Tensor arange(float start, float stop, float step = 1.f);

  // indexing
  Tensor operator[](const std::vector<int32_t> &idx) const;

  // transpose
  Tensor transpose(const std::vector<uint32_t> &axis = {}) const;
  inline static Tensor transpose(const Tensor &t, const std::vector<uint32_t> &axis = {}) {
    return t.transpose(axis);
  }

  // split
  std::vector<Tensor> split(uint32_t sections, const Axis &axis = 0) const;
  inline std::vector<Tensor> vsplit(uint32_t sections) const { return split(sections, 0); }
  inline std::vector<Tensor> hsplit(uint32_t sections) const { return split(sections, 1); }
  inline std::vector<Tensor> dsplit(uint32_t sections) const { return split(sections, 2); }

  std::vector<Tensor> split(const std::vector<uint32_t> &indices, const Axis &axis = 0) const;
  inline std::vector<Tensor> vsplit(const std::vector<uint32_t> &indices) { return split(indices, 0); }
  inline std::vector<Tensor> hsplit(const std::vector<uint32_t> &indices) { return split(indices, 1); }
  inline std::vector<Tensor> dsplit(const std::vector<uint32_t> &indices) { return split(indices, 2); }

  inline static std::vector<Tensor> split(const Tensor &t, uint32_t sections, const Axis &axis = 0) {
    return t.split(sections, axis);
  }
  inline static std::vector<Tensor> vsplit(const Tensor &t, uint32_t sections) { return t.split(sections, 0); }
  inline static std::vector<Tensor> hsplit(const Tensor &t, uint32_t sections) { return t.split(sections, 1); }
  inline static std::vector<Tensor> dsplit(const Tensor &t, uint32_t sections) { return t.split(sections, 2); }

  inline static std::vector<Tensor> split(const Tensor &t, const std::vector<uint32_t> &indices, const Axis &axis = 0) {
    return t.split(indices, axis);
  }
  inline static std::vector<Tensor> vsplit(const Tensor &t, const std::vector<uint32_t> &indices) { return t.split(indices, 0); }
  inline static std::vector<Tensor> hsplit(const Tensor &t, const std::vector<uint32_t> &indices) { return t.split(indices, 1); }
  inline static std::vector<Tensor> dsplit(const Tensor &t, const std::vector<uint32_t> &indices) { return t.split(indices, 2); }

  // concatenate
  static Tensor concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays);
  static Tensor concatenate(const std::vector<std::reference_wrapper<Tensor>> &arrays, const Axis &axis);

  // stack
  static Tensor stack(const std::vector<std::reference_wrapper<Tensor>> &arrays, const Axis &axis = 0);
  static Tensor vstack(const std::vector<std::reference_wrapper<Tensor>> &arrays);
  static Tensor hstack(const std::vector<std::reference_wrapper<Tensor>> &arrays);
  static Tensor dstack(const std::vector<std::reference_wrapper<Tensor>> &arrays);

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

  static Tensor sqrt(const Tensor &t);
  static Tensor tanh(const Tensor &t);
  static Tensor exp(const Tensor &t);

  Tensor sqrt() const { return Tensor::sqrt(*this); }
  Tensor tanh() const { return Tensor::tanh(*this); }
  Tensor exp() const { return Tensor::exp(*this); }

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

  inline float min() const { return Tensor::min(*this); };
  inline float max() const { return Tensor::max(*this); };
  inline float mean() const { return Tensor::mean(*this); };
  inline float sum() const { return Tensor::sum(*this); };
  inline float var() const { return Tensor::var(*this); };
  inline float argmin() const { return Tensor::argmin(*this); };
  inline float argmax() const { return Tensor::argmax(*this); };

  static Tensor min(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor max(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor mean(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor sum(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor var(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor argmin(const Tensor &t, const Axis &axis, bool keepDims = false);
  static Tensor argmax(const Tensor &t, const Axis &axis, bool keepDims = false);

  inline Tensor min(const Axis &axis, bool keepDims = false) const {
    return Tensor::min(*this, axis, keepDims);
  };
  inline Tensor max(const Axis &axis, bool keepDims = false) const {
    return Tensor::max(*this, axis, keepDims);
  };
  inline Tensor mean(const Axis &axis, bool keepDims = false) const {
    return Tensor::mean(*this, axis, keepDims);
  };
  inline Tensor sum(const Axis &axis, bool keepDims = false) const {
    return Tensor::sum(*this, axis, keepDims);
  };
  inline Tensor var(const Axis &axis, bool keepDims = false) const {
    return Tensor::var(*this, axis, keepDims);
  };
  inline Tensor argmin(const Axis &axis, bool keepDims = false) const {
    return Tensor::argmin(*this, axis, keepDims);
  };
  inline Tensor argmax(const Axis &axis, bool keepDims = false) const {
    return Tensor::argmax(*this, axis, keepDims);
  };

 protected:
  inline void initMeta();
  inline void initData();

  inline float &operator[](uint32_t idx) {
    return data_[idx];
  }

  inline const float &operator[](uint32_t idx) const {
    return data_[idx];
  }

  inline void traverse(UFunc &func, uint32_t start, uint32_t stride, uint32_t cnt) const;
  Tensor reduce(UFunc &func, uint32_t axis, bool keepDims = false) const;
  void splitAxis(std::vector<Tensor> &retTensors, std::vector<uint32_t> &splitIndices, uint32_t axis) const;

  static Tensor arraysConcat(const std::vector<std::reference_wrapper<Tensor>> &arrays, const Shape &retShape,
                             const std::vector<uint32_t> &concatIndices, uint32_t axis);
  static ShapeCompatible checkCompatible(const Shape &t0, const Shape &t1, Shape &retShape, uint32_t skipLast = 0);
  static bool checkShapeEqual(const std::vector<std::reference_wrapper<Tensor>> &arrays, uint32_t exceptAxis);
  static void error(const char *where, TensorError error);

 private:
  static float fastTanh(float x);

 protected:
  uint32_t dimCount_ = 0;
  uint32_t elemCount_ = 0;
  Shape shape_;
  std::vector<int32_t> strides_;
  std::vector<float> data_;
};

class TensorIter {
 public:
  explicit TensorIter(const Shape &shape);

  // get shape
  Shape shape();

  // reshape
  void reshape(const Shape &shape);

  // get size
  inline uint32_t size() const { return size_; }

  // get current coordinates
  inline const int32_t *coordinates() const { return coordinates_; };

  // return -1 if not available
  int32_t next();

  // reset to init states
  void reset();

  // broadcast to shape (no broadcast rules check)
  void broadcast(const Shape &shape);

  // transpose
  void transpose(const std::vector<uint32_t> &axis);

 protected:
  // reorder array
  static void reorder(int32_t *v, std::vector<uint32_t> const &order) {
    for (uint32_t s = 1, d; s < order.size(); ++s) {
      for (d = order[s]; d < s; d = order[d]);
      if (d == s) while (d = order[d], d != s) std::swap(v[s], v[d]);
    }
  }

 protected:
  int32_t ndM1_;
  int32_t size_;
  int32_t dimsM1_[TENSOR_MAX_DIMS];

  int32_t strides_[TENSOR_MAX_DIMS];
  int32_t backStrides_[TENSOR_MAX_DIMS];

  int32_t coordinates_[TENSOR_MAX_DIMS];
  int32_t index_;
  int32_t itCnt_;
};

}
