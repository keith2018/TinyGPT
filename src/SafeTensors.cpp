/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "SafeTensors.h"

#include <fstream>
#include <sstream>

#include "Utils/Logger.h"
#include "Utils/MMapUtils.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "util/PathUtils.h"

namespace tinygpt {

namespace tt = tinytorch;

constexpr const char* KeySafeTensorsMeta = "__metadata__";

std::string SafeTensors::toTypeString(tt::DType type) {
  switch (type) {
    case tt::DType::Float32:
      return "F32";
    case tt::DType::Float16:
      return "F16";
    case tt::DType::BFloat16:
      return "BF16";
    case tt::DType::Int32:
      return "I32";
    case tt::DType::Int64:
      return "I64";
    case tt::DType::Bool:
      return "BOOL";
    default:
      break;
  }

  LOGE("Unknown tt::DType: %s", tt::dtype::toString(type).c_str());
  ASSERT(false);
  return {};
}

tt::DType SafeTensors::fromTypeString(const std::string& s) {
  if (s == "F32") return tt::DType::Float32;
  if (s == "F16") return tt::DType::Float16;
  if (s == "BF16") return tt::DType::BFloat16;
  if (s == "I32") return tt::DType::Int32;
  if (s == "I64") return tt::DType::Int64;
  if (s == "BOOL") return tt::DType::Bool;

  LOGE("Unknown safeTensors dtype string: %s", s.c_str());
  ASSERT(false);
  return tt::DType::Float32;
}

bool SafeTensors::save(tt::nn::Module& module, const std::string& path) {
  auto namedStates = module.namedStates();

  rapidjson::Document headerDoc(rapidjson::kObjectType);
  auto& allocator = headerDoc.GetAllocator();

  size_t offset = 0;
  for (const auto& [name, tensor] : namedStates) {
    rapidjson::Value tensorInfo(rapidjson::kObjectType);

    rapidjson::Value shapeArr(rapidjson::kArrayType);
    for (auto dim : tensor->shape()) {
      shapeArr.PushBack(dim, allocator);
    }
    // shape
    tensorInfo.AddMember("shape", shapeArr, allocator);

    // dtype
    tensorInfo.AddMember("dtype", rapidjson::Value(toTypeString(tensor->dtype()).c_str(), allocator), allocator);

    // data_offsets
    size_t tensorSize = tensor->numel() * dtypeSize(tensor->dtype());
    rapidjson::Value offsets_arr(rapidjson::kArrayType);
    offsets_arr.PushBack(static_cast<uint64_t>(offset), allocator);
    offsets_arr.PushBack(static_cast<uint64_t>(offset + tensorSize), allocator);
    tensorInfo.AddMember("data_offsets", offsets_arr, allocator);

    headerDoc.AddMember(rapidjson::Value(name.c_str(), allocator), tensorInfo, allocator);
    offset += tensorSize;
  }

  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  headerDoc.Accept(writer);
  std::string headerStr = sb.GetString();

  size_t headerLen = headerStr.size();
  size_t alignedHeaderLen = ((headerLen + 7) / 8) * 8;
  headerStr.resize(alignedHeaderLen, ' ');

  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) {
    LOGE("Error open file: %s", path.c_str());
    return false;
  }
  uint64_t headerSize = alignedHeaderLen;
  ofs.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));
  ofs.write(headerStr.data(), static_cast<std::streamsize>(headerStr.size()));

  for (const auto& [name, tensor] : namedStates) {
    size_t tensorSize = tensor->numel() * dtypeSize(tensor->dtype());

    if (tensor->device().isCpu()) {
      ofs.write(static_cast<const char*>(tensor->dataPtr<>()), static_cast<std::streamsize>(tensorSize));
    } else {
      auto cpuTensor = tensor->to(tt::DeviceType::CPU);
      ofs.write(static_cast<const char*>(cpuTensor.dataPtr<>()), static_cast<std::streamsize>(tensorSize));
    }
  }
  ofs.close();
  return true;
}

bool SafeTensors::load(tt::nn::Module& module, const std::string& path, bool strict) {
  auto endsWith = [](const std::string& str, const std::string& suffix) {
    return suffix.size() <= str.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
  };

  if (endsWith(path, ".index.json")) {
    return loadMulti(module, path, strict);
  }

  if (endsWith(path, ".safetensors")) {
    return loadInternal(module, path, strict, {});
  }

  LOGE("Unknown file type: %s", path.c_str());
  return false;
}

bool SafeTensors::loadInternal(tt::nn::Module& module, const std::string& path, bool strict,
                               const ankerl::unordered_dense::set<std::string>& onlyKeys) {
  tt::MMappingResult mappingResult = tt::MMapUtils::mapFileForRead(path);
  if (!mappingResult.success) {
    LOGE("Error mapFileForRead: %s", path.c_str());
    return false;
  }

  void* fileMap = mappingResult.dataPtr;
  uint64_t headerSize = *static_cast<uint64_t*>(fileMap);
  const char* headerPtr = static_cast<const char*>(fileMap) + sizeof(uint64_t);
  std::string headerStr(headerPtr, headerSize);

  rapidjson::Document headerDoc;
  headerDoc.Parse(headerStr.c_str());

  ankerl::unordered_dense::map<std::string, tt::TensorPtr> name2tensor;
  for (const auto& [name, tensor] : module.namedStates()) {
    name2tensor[name] = tensor;
  }

  bool success = true;
  ankerl::unordered_dense::set<std::string> fileKeys;
  for (auto it = headerDoc.MemberBegin(); it != headerDoc.MemberEnd(); ++it) {
    std::string name = it->name.GetString();
    if (KeySafeTensorsMeta == name) {
      continue;
    }
    if (!onlyKeys.empty() && onlyKeys.count(name) == 0) {
      continue;
    }

    fileKeys.insert(name);
    const auto& info = it->value;

    auto iter = name2tensor.find(name);
    if (iter == name2tensor.end()) {
      LOGW("Unexpected key: %s", name.c_str());
      if (strict) {
        success = false;
      }
      continue;
    }
    tt::TensorPtr tensor = iter->second;

    // shape
    tt::SizeVector shape;
    for (auto& v : info["shape"].GetArray()) shape.pushBack(v.GetInt64());
    if (shape != tensor->shape()) {
      LOGE("shape not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }

    // dtype
    std::string dtype = info["dtype"].GetString();
    if (fromTypeString(dtype) != tensor->dtype()) {
      LOGE("dtype not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }

    // data_offsets
    size_t start = info["data_offsets"][0].GetUint64();
    size_t end = info["data_offsets"][1].GetUint64();
    size_t nbytes = end - start;
    size_t tensorSize = tensor->numel() * dtypeSize(tensor->dtype());
    if (nbytes != tensorSize) {
      LOGE("size not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }
    const void* dataPtr = static_cast<const char*>(fileMap) + sizeof(uint64_t) + headerSize + start;
    tt::Storage::copyOnDevice(tensor->dataPtr<>(), tensor->device(), dataPtr, tt::Device::cpu(),
                              static_cast<int64_t>(nbytes));
  }

  if (onlyKeys.empty()) {
    for (const auto& [name, tensor] : name2tensor) {
      if (!fileKeys.count(name)) {
        LOGW("Missing key: %s", name.c_str());
        if (strict) success = false;
      }
    }
  }

  tt::MMapUtils::unmapFile(mappingResult);
  return success;
}

bool SafeTensors::loadMulti(tt::nn::Module& module, const std::string& indexPath, bool strict) {
  std::ifstream ifs(indexPath);
  if (!ifs.is_open()) {
    LOGE("Error open index file: %s", indexPath.c_str());
    return false;
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string indexStr = buffer.str();

  rapidjson::Document indexDoc;
  indexDoc.Parse(indexStr.c_str());
  if (!indexDoc.IsObject()) {
    LOGE("Invalid index json: %s", indexPath.c_str());
    return false;
  }
  if (!indexDoc.HasMember("weight_map")) {
    LOGE("Index json missing weight_map");
    return false;
  }

  const auto& weightMap = indexDoc["weight_map"];
  if (!weightMap.IsObject()) {
    LOGE("'weight_map' is not object");
    return false;
  }

  ankerl::unordered_dense::map<std::string, std::vector<std::string>> shard2keys;
  for (auto it = weightMap.MemberBegin(); it != weightMap.MemberEnd(); ++it) {
    std::string tensorName = it->name.GetString();
    std::string shardFile = it->value.GetString();
    shard2keys[shardFile].push_back(tensorName);
  }

  bool success = true;
  std::string baseDir = PathUtils::getBaseDir(indexPath);
  for (const auto& [shardFile, keys] : shard2keys) {
    std::string shardPath = PathUtils::joinPath(baseDir, shardFile);

    ankerl::unordered_dense::set<std::string> keySet(keys.begin(), keys.end());
    if (!loadInternal(module, shardPath, false, keySet)) {
      LOGE("Failed to load shard: %s", shardPath.c_str());
      success = false;
      if (strict) {
        break;
      }
    }
  }
  return success;
}

}  // namespace tinygpt
