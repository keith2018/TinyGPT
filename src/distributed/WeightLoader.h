/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>
#include <vector>

#include "Tensor.h"
#include "ankerl/unordered_dense.h"

namespace tinygpt::distributed {

enum class ShardMode : uint8_t {
  REPLICATE = 0,      // not sharded, every rank owns full copy
  COLUMN = 1,         // shard along dim 0 of weight ([outFeatures, inFeatures])
  COLUMN_MERGED = 2,  // multi-segment column shard (gate+up etc.)
  ROW = 3,            // shard along dim 1 of weight
  QKV = 4,            // q/k/v segments, each sharded by head
};

struct ShardInfo {
  ShardMode mode = ShardMode::REPLICATE;
  std::vector<int64_t> partSizes;
};

class WeightLoader {
 public:
  static void tag(const tinytorch::Tensor& param, ShardInfo info);

  static const ShardInfo* lookup(const tinytorch::Tensor& param);

  static void clear();

 private:
  static ankerl::unordered_dense::map<const void*, ShardInfo>& registry();
};

}  // namespace tinygpt::distributed
