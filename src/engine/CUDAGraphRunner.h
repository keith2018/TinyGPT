/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>

#include "engine/ForwardContext.h"
#include "kernel/SamplerOps.h"
#include "model/GPTModel.h"

namespace tinygpt {

class CUDAGraphRunner {
 public:
  CUDAGraphRunner(GPTModel& model, tinytorch::Device device, int32_t batchSize = 1);
  ~CUDAGraphRunner();

  CUDAGraphRunner(const CUDAGraphRunner&) = delete;
  CUDAGraphRunner& operator=(const CUDAGraphRunner&) = delete;

  bool captured() const;

  void capture(ForwardContext& ctx, const tinytorch::Tensor& inputIds, tinytorch::Tensor& sampledHostBuf,
               void* sampledEvent, void* stream, const kernel::SamplingParams* samplingParams = nullptr);

  void replay(void* stream);
  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  GPTModel& model_;
  tinytorch::Device device_;
  int32_t batchSize_;
};

}  // namespace tinygpt
