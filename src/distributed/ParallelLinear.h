/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Communicator.h"
#include "WeightLoader.h"
#include "layer/Linear.h"

namespace tinygpt::distributed {

namespace tt = tinytorch;

class ColumnParallelLinear : public tt::nn::Linear {
 public:
  ColumnParallelLinear(int64_t inFeatures, int64_t outFeatures, bool bias = false, tt::Options options = {})
      : tt::nn::Linear(inFeatures, shardOut(outFeatures), bias, options), fullOutFeatures_(outFeatures) {
    ShardInfo info{ShardMode::COLUMN, {fullOutFeatures_}};
    WeightLoader::tag(weight_, info);
    if (useBias_) {
      WeightLoader::tag(bias_, info);
    }
  }

  ColumnParallelLinear(ColumnParallelLinear&& other) noexcept
      : tt::nn::Linear(std::move(other)), fullOutFeatures_(other.fullOutFeatures_) {}

  ColumnParallelLinear& operator=(ColumnParallelLinear&&) = delete;
  ColumnParallelLinear(const ColumnParallelLinear&) = delete;
  ColumnParallelLinear& operator=(const ColumnParallelLinear&) = delete;

  int64_t fullOutFeatures() const { return fullOutFeatures_; }

 private:
  static int64_t shardOut(int64_t full) {
    int ws = Communicator::tp().worldSize();
    ASSERT(full % ws == 0);
    return full / ws;
  }

  int64_t fullOutFeatures_;
};

class MergedColumnParallelLinear : public tt::nn::MergedLinear {
 public:
  MergedColumnParallelLinear(int64_t inputSize, tt::IntArrayView outputSizes, bool bias = false,
                             tt::Options options = {})
      : tt::nn::MergedLinear(inputSize, shardSizes(outputSizes), bias, options),
        fullOutputSizes_(outputSizes.begin(), outputSizes.end()) {
    auto& wSegs = weightSegments();
    auto& bSegs = biasSegments();
    for (size_t i = 0; i < fullOutputSizes_.size(); i++) {
      ShardInfo info{ShardMode::COLUMN, {fullOutputSizes_[i]}};
      WeightLoader::tag(wSegs[i], info);
      if (useBias_) {
        WeightLoader::tag(bSegs[i], info);
      }
    }
  }

  MergedColumnParallelLinear(MergedColumnParallelLinear&& other) noexcept
      : tt::nn::MergedLinear(std::move(other)), fullOutputSizes_(std::move(other.fullOutputSizes_)) {}

  MergedColumnParallelLinear& operator=(MergedColumnParallelLinear&&) = delete;
  MergedColumnParallelLinear(const MergedColumnParallelLinear&) = delete;
  MergedColumnParallelLinear& operator=(const MergedColumnParallelLinear&) = delete;

 private:
  static std::vector<int64_t> shardSizes(tt::IntArrayView sizes) {
    int ws = Communicator::tp().worldSize();
    std::vector<int64_t> out;
    out.reserve(sizes.size());
    for (auto s : sizes) {
      ASSERT(s % ws == 0);
      out.push_back(s / ws);
    }
    return out;
  }

  std::vector<int64_t> fullOutputSizes_;
};

class RowParallelLinear : public tt::nn::GemvLinear {
 public:
  RowParallelLinear(int64_t inFeatures, int64_t outFeatures, bool bias = false, tt::Options options = {})
      : tt::nn::GemvLinear(shardIn(inFeatures), outFeatures, bias, options), fullInFeatures_(inFeatures) {
    ShardInfo info{ShardMode::ROW, {fullInFeatures_}};
    WeightLoader::tag(weight_, info);
    ASSERT(!useBias_ && "RowParallelLinear: bias not supported (would be added worldSize times)");
  }

  RowParallelLinear(RowParallelLinear&& other) noexcept
      : tt::nn::GemvLinear(std::move(other)), fullInFeatures_(other.fullInFeatures_) {}

  RowParallelLinear& operator=(RowParallelLinear&&) = delete;
  RowParallelLinear(const RowParallelLinear&) = delete;
  RowParallelLinear& operator=(const RowParallelLinear&) = delete;

  tt::Tensor forward(const tt::Tensor& input) override {
    auto out = tt::nn::GemvLinear::forward(input);
    Communicator::tp().allReduceSum(out);
    return out;
  }

  int64_t fullInFeatures() const { return fullInFeatures_; }

 private:
  static int64_t shardIn(int64_t full) {
    int ws = Communicator::tp().worldSize();
    ASSERT(full % ws == 0);
    return full / ws;
  }

  int64_t fullInFeatures_;
};

class VocabParallelLinear : public tt::nn::LmHeadLinear {
 public:
  VocabParallelLinear(int64_t inFeatures, int64_t outFeatures, bool bias = false, tt::Options options = {})
      : tt::nn::LmHeadLinear(inFeatures, shardOut(outFeatures), bias, options), fullOutFeatures_(outFeatures) {
    ShardInfo info{ShardMode::COLUMN, {fullOutFeatures_}};
    WeightLoader::tag(weight_, info);
    if (useBias_) {
      WeightLoader::tag(bias_, info);
    }
  }

  VocabParallelLinear(VocabParallelLinear&& other) noexcept
      : tt::nn::LmHeadLinear(std::move(other)), fullOutFeatures_(other.fullOutFeatures_) {}

  VocabParallelLinear& operator=(VocabParallelLinear&&) = delete;
  VocabParallelLinear(const VocabParallelLinear&) = delete;
  VocabParallelLinear& operator=(const VocabParallelLinear&) = delete;

  tt::Tensor forward(const tt::Tensor& input) override {
    // local: [batch, localVocab] (gemv fast-path applied if M == 1)
    auto local = tt::nn::LmHeadLinear::forward(input);
    auto& comm = Communicator::tp();
    if (!comm.enabled()) {
      return local;
    }
    return gatherVocab(local);
  }

  int64_t fullOutFeatures() const { return fullOutFeatures_; }

 private:
  static int64_t shardOut(int64_t full) {
    int ws = Communicator::tp().worldSize();
    ASSERT(full % ws == 0);
    return full / ws;
  }

  // [batch, localVocab] -> [batch, fullVocab] via allGather along vocab dim
  static tt::Tensor gatherVocab(const tt::Tensor& local) {
    auto& comm = Communicator::tp();
    const int ws = comm.worldSize();
    const int64_t batch = local.size(0);
    const int64_t localVocab = local.size(1);

    auto opts = tt::Options(local.device(), local.dtype()).noGrad();
    // [ws, batch, localVocab] contiguous; per-rank views are contiguous slices
    auto fullBuf = tt::Tensor({static_cast<int64_t>(ws), batch, localVocab}, opts);
    std::vector<tt::Tensor> rankOuts;
    rankOuts.reserve(ws);
    for (int r = 0; r < ws; r++) {
      rankOuts.push_back(fullBuf.narrow(0, r, 1).view({batch, localVocab}));
    }

    std::vector<std::vector<tt::Tensor>> outputs = {rankOuts};
    std::vector<tt::Tensor> inputs = {local};
    auto work = comm.pg()->allGather(outputs, inputs, {});
    if (work) {
      work->wait();
    }

    // [ws, batch, localVocab] -> [batch, ws, localVocab] -> [batch, ws*localVocab]
    return fullBuf.permute({1, 0, 2}).view({batch, static_cast<int64_t>(ws) * localVocab});
  }

  int64_t fullOutFeatures_;
};

}  // namespace tinygpt::distributed
