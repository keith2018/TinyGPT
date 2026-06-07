/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "WeightLoader.h"

#include <mutex>

namespace tinygpt::distributed {

namespace {
std::mutex& registryMutex() {
  static std::mutex m;
  return m;
}
}  // namespace

ankerl::unordered_dense::map<const void*, ShardInfo>& WeightLoader::registry() {
  static ankerl::unordered_dense::map<const void*, ShardInfo> r;
  return r;
}

void WeightLoader::tag(const tinytorch::Tensor& param, ShardInfo info) {
  if (!param.defined()) {
    return;
  }
  std::lock_guard<std::mutex> lock(registryMutex());
  registry()[param.dataPtr<>()] = std::move(info);
}

const ShardInfo* WeightLoader::lookup(const tinytorch::Tensor& param) {
  if (!param.defined()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(registryMutex());
  auto& r = registry();
  auto it = r.find(param.dataPtr<>());
  return it == r.end() ? nullptr : &it->second;
}

void WeightLoader::clear() {
  std::lock_guard<std::mutex> lock(registryMutex());
  registry().clear();
}

}  // namespace tinygpt::distributed
