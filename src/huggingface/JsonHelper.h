/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "rapidjson/document.h"

namespace tinygpt::json {

template <typename T>
T getJsonValue(const rapidjson::Value& obj, const char* key, const T& defaultValue) {
  if (!(obj.IsObject() && obj.HasMember(key))) {
    return defaultValue;
  }
  const auto& val = obj[key];

  if constexpr (std::is_same_v<T, bool>) {
    return val.IsBool() ? val.GetBool() : defaultValue;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return val.IsInt() ? val.GetInt() : defaultValue;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return val.IsInt64() ? val.GetInt64() : defaultValue;
  } else if constexpr (std::is_same_v<T, float>) {
    return val.IsNumber() ? val.GetFloat() : defaultValue;
  } else if constexpr (std::is_same_v<T, std::string>) {
    return val.IsString() ? std::string(val.GetString(), val.GetStringLength()) : defaultValue;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type");
  }
  return defaultValue;
}

}  // namespace tinygpt::json