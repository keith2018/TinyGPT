/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <mutex>

namespace TinyGPT {

#define LOGI(...) TinyGPT::Logger::log(TinyGPT::LOG_INFO,     __FILE__, __LINE__, __VA_ARGS__)
#define LOGD(...) TinyGPT::Logger::log(TinyGPT::LOG_DEBUG,    __FILE__, __LINE__, __VA_ARGS__)
#define LOGW(...) TinyGPT::Logger::log(TinyGPT::LOG_WARNING,  __FILE__, __LINE__, __VA_ARGS__)
#define LOGE(...) TinyGPT::Logger::log(TinyGPT::LOG_ERROR,    __FILE__, __LINE__, __VA_ARGS__)

static constexpr int MAX_LOG_LENGTH = 1024;
typedef void (*LogFunc)(void *context, int level, const char *msg);

enum LogLevel {
  LOG_INFO, LOG_DEBUG, LOG_WARNING, LOG_ERROR,
};

class Logger {
 public:
  static void log(LogLevel level, const char *file, int line, const char *message, ...);

 private:
  static void *logContext_;
  static LogFunc logFunc_;
  static LogLevel minLevel_;

  static char buf_[MAX_LOG_LENGTH];
  static std::mutex mutex_;
};

}
