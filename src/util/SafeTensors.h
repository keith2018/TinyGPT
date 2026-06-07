/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"
#include "ankerl/unordered_dense.h"

namespace tinygpt {

namespace distributed {
struct ShardInfo;
}

class SafeTensors {
 public:
  static bool save(tinytorch::nn::Module& module, const std::string& path);
  static bool load(tinytorch::nn::Module& module, const std::string& path, bool strict = true);

 private:
  static std::string toTypeString(tinytorch::DType type);
  static tinytorch::DType fromTypeString(const std::string& s);

  static bool loadInternal(tinytorch::nn::Module& module, const std::string& path, bool strict,
                           const ankerl::unordered_dense::set<std::string>& onlyKeys);
  static bool loadMulti(tinytorch::nn::Module& module, const std::string& indexPath, bool strict = true);

  static bool loadSharded(tinytorch::Tensor& tensor, const distributed::ShardInfo& shard,
                          const tinytorch::SizeVector& fileShape, const void* baseDataPtr, uint64_t fileBytes,
                          size_t dtSize, int rank, int worldSize, const std::string& name);
};

}  // namespace tinygpt
