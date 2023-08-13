/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#define GPT_ENCODER_JSON "encoder.json"
#define GPT_VOCAB_BPE "vocab.bpe"
#define FILE_SEP "/"

namespace TinyGPT {

class StringUtils {
 public:
  static std::wstring utf82wstring(const std::string &str);
  static std::string wstring2utf8(const std::wstring &str);
  static std::vector<std::wstring> split(const std::wstring &s, wchar_t delim);
};

using wstring_pair = std::pair<std::wstring, std::wstring>;
struct hash_wstring_pair {
  size_t operator()(const std::pair<std::wstring, std::wstring> &p) const {
    std::size_t seed = 0;
    std::hash<std::wstring> hasher;
    seed ^= hasher(p.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hasher(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

/**
 * Ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
 */
class Encoder {
 public:
  Encoder() = default;
  Encoder(const std::unordered_map<std::wstring, int32_t> &encoder,
          const std::vector<wstring_pair> &bpeMerges);

  std::wstring bpe(const std::wstring &token);
  std::vector<int32_t> encode(const std::string &text);
  std::string decode(const std::vector<int32_t> &tokens);

  static Encoder getEncoder(const std::string &modelsDir);

 private:
  static std::vector<wstring_pair> getPairs(const std::vector<std::wstring> &word);
  static std::unordered_map<int32_t, wchar_t> bytesToUnicode();

  static std::unordered_map<std::wstring, int32_t> loadEncoderMap(std::istream &in);
  static std::vector<wstring_pair> loadVocabBpe(std::istream &in);

 private:
  std::unordered_map<std::wstring, int32_t> encoder_;
  std::unordered_map<int32_t, std::wstring> decoder_;
  std::unordered_map<int32_t, wchar_t> byteEncoder_;
  std::unordered_map<wchar_t, int32_t> byteDecoder_;
  std::unordered_map<wstring_pair, int32_t, hash_wstring_pair> bpeRanks_;
  std::unordered_map<std::wstring, std::wstring> cache_;
};

}
