/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <fstream>
#include <string>

namespace tinygpt {

class PathUtils {
 public:
  static bool fileExists(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    return file.good();
  }

  static std::string getBaseDir(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
      return path.substr(0, pos);
    }
    return ".";
  }

  static std::string joinPath(const std::string& dir, const std::string& file) {
    if (dir.empty() || dir == ".") {
      return file;
    }
    char lastChar = dir.back();
    if (lastChar == '/' || lastChar == '\\') {
      return dir + file;
    }
    return dir + "/" + file;
  }
};

}  // namespace tinygpt