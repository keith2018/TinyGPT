/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <fstream>

#include "Utils/Logger.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

namespace tinygpt {

class FileUtils {
 public:
  static std::string readText(std::istream& in) {
    in.seekg(0, std::ios::end);
    auto size = (size_t)in.tellg();
    if (size <= 0) {
      LOGE("file size invalid: %lld", size);
      return {};
    }

    std::string fileStr(size + 1, 0);
    in.seekg(0, std::ios::beg);
    in.read(&fileStr[0], (std::streamsize)size);

    return fileStr;
  }

  static std::string readText(const char* path) {
    std::fstream in(path, std::ios::in);
    if (!in.is_open()) {
      LOGE("open file failed: %s", path);
      return {};
    }

    return readText(in);
  }

  static rapidjson::Document parseJson(const std::string& str) {
    rapidjson::Document doc;
    rapidjson::ParseResult ok = doc.Parse(str.c_str());
    if (!ok) {
      LOGE("parse file error: %s (%zu)", rapidjson::GetParseError_En(ok.Code()), ok.Offset());
      doc.SetNull();
    }
    return doc;
  }

  static rapidjson::Document parseJson(const char* path) { return parseJson(readText(path)); }

  static rapidjson::Document parseJson(std::istream& in) { return parseJson(readText(in)); }
};

}  // namespace tinygpt