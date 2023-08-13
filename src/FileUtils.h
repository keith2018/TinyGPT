/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Logger.h"
#include "json11.hpp"

#include <fstream>
#include <sstream>

namespace TinyGPT {

class FileUtils {
 public:
  static std::string readText(std::istream &in) {
    in.seekg(0, std::ios::end);
    auto size = (size_t) in.tellg();
    if (size <= 0) {
      LOGE("file size invalid: %lld", size);
      return {};
    }

    std::string fileStr(size + 1, 0);
    in.seekg(0, std::ios::beg);
    in.read(&fileStr[0], (std::streamsize) size);

    return fileStr;
  }

  static std::string readText(const char *path) {
    std::fstream in(path, std::ios::in);
    if (!in.is_open()) {
      LOGE("open file failed: %s", path);
      return {};
    }

    return readText(in);
  }

  static json11::Json parseJson(const std::string &str) {
    std::string err;
    auto ret = json11::Json::parse(str.c_str(), err);
    if (ret.is_null()) {
      LOGE("parse file error: %s", err.c_str());
      return {};
    }

    return ret;
  }

  static json11::Json parseJson(const char *path) {
    return parseJson(readText(path));
  }

  static json11::Json parseJson(std::istream &in) {
    return parseJson(readText(in));
  }
};

}
