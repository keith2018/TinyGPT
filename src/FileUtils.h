/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>

#include "rapidjson/document.h"

namespace tinygpt {

struct FileMappingResult {
  void* dataPtr = nullptr;
  size_t fileSize = 0;
#ifdef _WIN32
  void* hFile = nullptr;
  void* hMap = nullptr;
#else
  int fd = -1;
#endif
  bool success = false;
};

class FileUtils {
 public:
  static std::string readText(std::istream& in);

  static std::string readText(const char* path);

  static FileMappingResult mapFileForRead(const std::string& path);

  static void unmapFile(FileMappingResult& mappingResult);

  static rapidjson::Document parseJson(const std::string& str);

  static rapidjson::Document parseJson(const char* path) { return parseJson(readText(path)); }

  static rapidjson::Document parseJson(std::istream& in) { return parseJson(readText(in)); }
};

}  // namespace tinygpt