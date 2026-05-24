/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinygpt {

class PagedKVCache;

struct ForwardContext {
  PagedKVCache* pagedCache = nullptr;

  tinytorch::Tensor positions;    // [totalTokens] int64
  tinytorch::Tensor cuSeqLensQ;   // [batchSize + 1] int32
  tinytorch::Tensor cuSeqLensKV;  // [batchSize + 1] int32
  tinytorch::Tensor slotMapping;  // [totalTokens] int32
  tinytorch::Tensor blockTable;   // [batchSize, maxBlocksPerSeq] int32

  int32_t maxSeqLenQ = 0;
  int32_t maxSeqLenKV = 0;
  int32_t pageSize = 0;
  int32_t maxBlocksPerSeq = 0;

  float* tmpO = nullptr;
  float* tmpLse = nullptr;

  // [batchSize] int64 — indices of last token per sequence for selective lm_head
  tinytorch::Tensor lastTokenIndices;

  static ForwardContext* current();
  static ForwardContext* setCurrent(ForwardContext* ctx);
};

class ForwardContextGuard {
 public:
  explicit ForwardContextGuard(ForwardContext* ctx) : prev_(ForwardContext::setCurrent(ctx)) {}
  ~ForwardContextGuard() { ForwardContext::setCurrent(prev_); }

  ForwardContextGuard(const ForwardContextGuard&) = delete;
  ForwardContextGuard& operator=(const ForwardContextGuard&) = delete;
  ForwardContextGuard(ForwardContextGuard&&) = delete;
  ForwardContextGuard& operator=(ForwardContextGuard&&) = delete;

 private:
  ForwardContext* prev_;
};

}  // namespace tinygpt
