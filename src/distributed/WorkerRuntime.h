/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinygpt {
class GPTModel;
class PagedKVCache;
}  // namespace tinygpt

namespace tinygpt::distributed {

class WorkerRuntime {
 public:
  WorkerRuntime(GPTModel& model, PagedKVCache& cache, int32_t maxBatchTokens);

  void run();

 private:
  void allocateMetaBuffers();

  GPTModel& model_;
  PagedKVCache& cache_;
  int32_t maxBatchTokens_;
  tinytorch::Device device_;

  tinytorch::Tensor metaDevI64_;
  tinytorch::Tensor metaDevI32_;
  tinytorch::Tensor tpHeaderDev_;
  tinytorch::Tensor tpHeaderHost_;

  tinytorch::Tensor splitKvO_;
  tinytorch::Tensor splitKvLse_;
};

}  // namespace tinygpt::distributed
