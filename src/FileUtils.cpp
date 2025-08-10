/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "FileUtils.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <fstream>

#include "Utils/Logger.h"
#include "rapidjson/error/en.h"

namespace tinygpt {

std::string FileUtils::readText(std::istream& in) {
  in.seekg(0, std::ios::end);
  auto size = static_cast<size_t>(in.tellg());
  if (size <= 0) {
    LOGE("file size invalid: %lld", size);
    return {};
  }

  std::string fileStr(size + 1, 0);
  in.seekg(0, std::ios::beg);
  in.read(&fileStr[0], static_cast<std::streamsize>(size));

  return fileStr;
}

std::string FileUtils::readText(const char* path) {
  std::fstream in(path, std::ios::in);
  if (!in.is_open()) {
    LOGE("open file failed: %s", path);
    return {};
  }

  return readText(in);
}

FileMappingResult FileUtils::mapFileForRead(const std::string& path) {
  FileMappingResult result;

#ifdef _WIN32
  result.hFile = INVALID_HANDLE_VALUE;  // init
  result.hFile =
      CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (result.hFile == INVALID_HANDLE_VALUE) {
    LOGE("Error open file: %s, GetLastError: %lu", path.c_str(), GetLastError());
    return result;
  }

  LARGE_INTEGER liFileSize;
  if (!GetFileSizeEx(result.hFile, &liFileSize)) {
    LOGE("GetFileSizeEx failed for file: %s, GetLastError: %lu", path.c_str(), GetLastError());
    CloseHandle(result.hFile);
    return result;
  }
  result.fileSize = static_cast<size_t>(liFileSize.QuadPart);

  result.hMap = CreateFileMapping(result.hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (result.hMap == nullptr) {
    LOGE("CreateFileMapping failed, GetLastError: %lu", GetLastError());
    CloseHandle(result.hFile);
    return result;
  }

  result.dataPtr = MapViewOfFile(result.hMap, FILE_MAP_READ, 0, 0, 0);
  if (result.dataPtr == nullptr) {
    LOGE("MapViewOfFile failed, GetLastError: %lu", GetLastError());
    CloseHandle(result.hMap);
    CloseHandle(result.hFile);
    return result;
  }
  result.success = true;

#else
  result.fd = open(path.c_str(), O_RDONLY);
  if (result.fd < 0) {
    LOGE("Error open file: %s", path.c_str());
    return result;
  }

  struct stat st{};
  if (fstat(result.fd, &st) < 0) {
    close(result.fd);
    LOGE("fstat failed: %s", path.c_str());
    return result;
  }
  result.fileSize = st.st_size;
  result.dataPtr = mmap(nullptr, result.fileSize, PROT_READ, MAP_PRIVATE, result.fd, 0);
  if (result.dataPtr == MAP_FAILED) {
    close(result.fd);
    LOGE("mmap failed");
    return result;
  }
  result.success = true;
#endif
  return result;
}

void FileUtils::unmapFile(FileMappingResult& mappingResult) {
#ifdef _WIN32
  if (mappingResult.dataPtr != nullptr) {
    UnmapViewOfFile(mappingResult.dataPtr);
  }
  if (mappingResult.hMap != nullptr) {
    CloseHandle(mappingResult.hMap);
  }
  if (mappingResult.hFile != INVALID_HANDLE_VALUE) {
    CloseHandle(mappingResult.hFile);
  }
#else
  if (mappingResult.dataPtr != MAP_FAILED && mappingResult.dataPtr != nullptr) {
    munmap(mappingResult.dataPtr, mappingResult.fileSize);
  }
  if (mappingResult.fd != -1) {
    close(mappingResult.fd);
  }
#endif
  mappingResult.dataPtr = nullptr;
  mappingResult.fileSize = 0;
#ifdef _WIN32
  mappingResult.hFile = INVALID_HANDLE_VALUE;
  mappingResult.hMap = nullptr;
#else
  mappingResult.fd = -1;
#endif
  mappingResult.success = false;
}

rapidjson::Document FileUtils::parseJson(const std::string& str) {
  rapidjson::Document doc;
  rapidjson::ParseResult ok = doc.Parse(str.c_str());
  if (!ok) {
    LOGE("parse file error: %s (%zu)", rapidjson::GetParseError_En(ok.Code()), ok.Offset());
    doc.SetNull();
  }
  return doc;
}

}  // namespace tinygpt