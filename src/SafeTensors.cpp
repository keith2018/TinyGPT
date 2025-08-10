/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "SafeTensors.h"

#include <fstream>

#include "FileUtils.h"
#include "Utils/Logger.h"
#include "ankerl/unordered_dense.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace tinygpt {

using namespace tinytorch;

std::string SafeTensors::toTypeString(DType type) {
  switch (type) {
    case DType::Float32:
      return "F32";
    case DType::Float16:
      return "F16";
    case DType::BFloat16:
      return "BF16";
    case DType::Int32:
      return "I32";
    case DType::Int64:
      return "I64";
    case DType::Bool:
      return "BOOL";
    default:
      break;
  }

  LOGE("Unknown DType: %s", dtype::toString(type).c_str());
  ASSERT(false);
  return {};
}

DType SafeTensors::fromTypeString(const std::string& s) {
  if (s == "F32") return DType::Float32;
  if (s == "F16") return DType::Float16;
  if (s == "BF16") return DType::BFloat16;
  if (s == "I32") return DType::Int32;
  if (s == "I64") return DType::Int64;
  if (s == "BOOL") return DType::Bool;

  LOGE("Unknown safeTensors dtype string: %s", s.c_str());
  ASSERT(false);
  return DType::Float32;
}

bool SafeTensors::save(nn::Module& module, const std::string& path) {
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
    ofs.write(static_cast<const char*>(tensor->dataPtr<>()), static_cast<std::streamsize>(tensorSize));
  }
  ofs.close();
  return true;
}

bool SafeTensors::load(nn::Module& module, const std::string& path) {
  FileMappingResult mappingResult = FileUtils::mapFileForRead(path);
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

  ankerl::unordered_dense::map<std::string, TensorPtr> name2tensor;
  for (const auto& [name, tensor] : module.namedStates()) {
    name2tensor[name] = tensor;
  }

  bool success = true;
  for (auto it = headerDoc.MemberBegin(); it != headerDoc.MemberEnd(); ++it) {
    std::string name = it->name.GetString();
    const auto& info = it->value;

    auto iter = name2tensor.find(name);
    if (iter == name2tensor.end()) {
      LOGW("tensor not found: %s", name.c_str());
      continue;
    }
    TensorPtr tensor = iter->second;

    // shape
    SizeVector shape;
    for (auto& v : info["shape"].GetArray()) shape.pushBack(v.GetInt64());
    if (shape != tensor->shape()) {
      LOGE("shape not equal for tensor: %s", name.c_str());
      success = false;
      break;
    }

    // dtype
    std::string dtype = info["dtype"].GetString();
    if (fromTypeString(dtype) != tensor->dtype()) {
      LOGE("dtype not equal for tensor: %s", name.c_str());
      success = false;
      break;
    }

    // data_offsets
    size_t start = info["data_offsets"][0].GetUint64();
    size_t end = info["data_offsets"][1].GetUint64();
    size_t nbytes = end - start;
    size_t tensorSize = tensor->numel() * dtypeSize(tensor->dtype());
    if (nbytes != tensorSize) {
      LOGE("size not equal for tensor: %s", name.c_str());
      success = false;
      break;
    }
    const void* dataPtr = static_cast<const char*>(fileMap) + sizeof(uint64_t) + headerSize + start;
    Storage::copyOnDevice(tensor->dataPtr<>(), tensor->device(), dataPtr, Device::cpu(), static_cast<int64_t>(nbytes));
  }

  FileUtils::unmapFile(mappingResult);
  return success;
}

}  // namespace tinygpt
