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
#include "distributed/Communicator.h"
#include "distributed/WeightLoader.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "util/FileUtils.h"

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

  LOGE("Unknown tt::DType: %s", tt::dtypeToString(type));
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

  uint64_t offset = 0;
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
    uint64_t tensorSize = static_cast<uint64_t>(tensor->numel()) * dtypeSize(tensor->dtype());
    rapidjson::Value offsets_arr(rapidjson::kArrayType);
    offsets_arr.PushBack(offset, allocator);
    offsets_arr.PushBack(offset + tensorSize, allocator);
    tensorInfo.AddMember("data_offsets", offsets_arr, allocator);

    headerDoc.AddMember(rapidjson::Value(name.c_str(), allocator), tensorInfo, allocator);
    offset += tensorSize;
  }

  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  headerDoc.Accept(writer);
  std::string headerStr = sb.GetString();

  uint64_t headerLen = headerStr.size();
  uint64_t alignedHeaderLen = ((headerLen + 7) / 8) * 8;
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
    uint64_t tensorSize = static_cast<uint64_t>(tensor->numel()) * dtypeSize(tensor->dtype());

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

    // shape from file (full, unsharded)
    tt::SizeVector fileShape;
    for (auto& v : info["shape"].GetArray()) {
      fileShape.pushBack(v.GetInt64());
    }

    // dtype
    std::string dtype = info["dtype"].GetString();
    if (fromTypeString(dtype) != tensor->dtype()) {
      LOGE("dtype not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }

    uint64_t fileStart = info["data_offsets"][0].GetUint64();
    uint64_t fileEnd = info["data_offsets"][1].GetUint64();
    uint64_t fileBytes = fileEnd - fileStart;
    const auto* baseDataPtr = static_cast<const char*>(fileMap) + sizeof(uint64_t) + headerSize + fileStart;
    const size_t dtSize = dtypeSize(tensor->dtype());

    // sharded path
    const auto* shard = distributed::WeightLoader::lookup(*tensor);
    const int worldSize = distributed::Communicator::tp().worldSize();
    if (shard && worldSize > 1 && shard->mode != distributed::ShardMode::REPLICATE) {
      const int rank = distributed::Communicator::tp().rank();
      if (!loadSharded(*tensor, *shard, fileShape, baseDataPtr, fileBytes, dtSize, rank, worldSize, name)) {
        success = false;
      }
      continue;
    }

    // replicated path
    if (fileShape != tensor->shape()) {
      LOGE("shape not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }
    uint64_t tensorSize = static_cast<uint64_t>(tensor->numel()) * dtSize;
    if (fileBytes != tensorSize) {
      LOGE("size not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }
    tt::Storage::copyOnDevice(tensor->dataPtr<>(), tensor->device(), baseDataPtr, tt::Device::cpu(),
                              static_cast<int64_t>(fileBytes));
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

bool SafeTensors::loadSharded(tt::Tensor& tensor, const distributed::ShardInfo& shard, const tt::SizeVector& fileShape,
                              const void* baseDataPtr, uint64_t fileBytes, size_t dtSize, int rank, int worldSize,
                              const std::string& name) {
  using distributed::ShardMode;

  if (shard.mode == ShardMode::COLUMN) {
    // file shape [fullOut, ...], local shape [fullOut/ws, ...]; contiguous slice
    const int64_t fullOut = shard.partSizes.empty() ? fileShape[0] : shard.partSizes[0];
    if (fileShape[0] != fullOut || fullOut % worldSize != 0) {
      LOGE("shard COLUMN size mismatch for %s: fullOut=%lld file[0]=%lld ws=%d", name.c_str(),
           static_cast<long long>(fullOut), static_cast<long long>(fileShape[0]), worldSize);
      return false;
    }
    const int64_t localOut = fullOut / worldSize;
    int64_t innerCount = 1;
    for (size_t i = 1; i < fileShape.size(); i++) {
      innerCount *= fileShape[i];
    }

    const uint64_t shardBytes = static_cast<uint64_t>(localOut) * innerCount * dtSize;
    if (shardBytes != static_cast<uint64_t>(tensor.numel()) * dtSize) {
      LOGE("shard COLUMN local size mismatch for %s", name.c_str());
      return false;
    }
    const auto* srcPtr = static_cast<const char*>(baseDataPtr) + rank * shardBytes;
    tt::Storage::copyOnDevice(tensor.dataPtr<>(), tensor.device(), srcPtr, tt::Device::cpu(),
                              static_cast<int64_t>(shardBytes));
    return true;
  }

  if (shard.mode == ShardMode::ROW) {
    // file shape [outFeatures, fullIn], local shape [outFeatures, fullIn/ws]; row-by-row copy
    if (fileShape.size() != 2) {
      LOGE("shard ROW expects 2-D weight, got %zu-D for %s", fileShape.size(), name.c_str());
      return false;
    }
    const int64_t outFeatures = fileShape[0];
    const int64_t fullIn = fileShape[1];
    if (fullIn % worldSize != 0) {
      LOGE("shard ROW fullIn=%lld not divisible by ws=%d for %s", static_cast<long long>(fullIn), worldSize,
           name.c_str());
      return false;
    }
    const int64_t localIn = fullIn / worldSize;
    if (outFeatures != tensor.shape()[0] || localIn != tensor.shape()[1]) {
      LOGE("shard ROW shape mismatch for %s", name.c_str());
      return false;
    }
    const uint64_t rowBytesFile = static_cast<uint64_t>(fullIn) * dtSize;
    const uint64_t rowBytesLocal = static_cast<uint64_t>(localIn) * dtSize;
    const uint64_t srcRowOff = static_cast<uint64_t>(rank) * rowBytesLocal;
    auto* dstBase = static_cast<char*>(tensor.dataPtr<>());
    for (int64_t r = 0; r < outFeatures; r++) {
      const auto* srcPtr = static_cast<const char*>(baseDataPtr) + r * rowBytesFile + srcRowOff;
      tt::Storage::copyOnDevice(dstBase + r * rowBytesLocal, tensor.device(), srcPtr, tt::Device::cpu(),
                                static_cast<int64_t>(rowBytesLocal));
    }
    UNUSED(fileBytes);
    return true;
  }

  LOGE("unsupported shard mode for tensor: %s", name.c_str());
  return false;
}

bool SafeTensors::loadMulti(tt::nn::Module& module, const std::string& indexPath, bool strict) {
  std::ifstream ifs(indexPath, std::ios::binary);
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
  std::string indexDir = fileutil::baseDir(indexPath);
  for (const auto& [shardFile, keys] : shard2keys) {
    std::string shardPath = fileutil::join(indexDir, shardFile);

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
