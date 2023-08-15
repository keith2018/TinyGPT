/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tokenizer.h"
#include "Logger.h"
#include "Timer.h"
#include "FileUtils.h"
#include "re2/re2.h"

#include <fstream>
#include <sstream>
#include <locale>
#include <codecvt>

namespace TinyGPT {

RE2 gEncoderPat_("('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
                 "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)");

std::wstring StringUtils::utf82wstring(const std::string &str) {
  static std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  return conv.from_bytes(str);
}

std::string StringUtils::wstring2utf8(const std::wstring &str) {
  static std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  return conv.to_bytes(str);
}

std::vector<std::wstring> StringUtils::split(const std::wstring &s, wchar_t delim) {
  std::wstringstream ss(s);
  std::wstring item;
  std::vector<std::wstring> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

Encoder::Encoder(const std::unordered_map<std::wstring, int32_t> &encoder,
                 const std::vector<wstring_pair> &bpeMerges)
    : Encoder() {
  encoder_ = encoder;
  for (auto &kv : encoder_) {
    decoder_.insert({kv.second, kv.first});
  }
  byteEncoder_ = bytesToUnicode();
  for (auto &kv : byteEncoder_) {
    byteDecoder_.insert({kv.second, kv.first});
  }
  for (size_t idx = 0; idx < bpeMerges.size(); idx++) {
    bpeRanks_.insert({bpeMerges[idx], (int32_t) idx});
  }
  cache_.clear();
}

std::wstring Encoder::bpe(const std::wstring &token) {
  auto cacheItem = cache_.find(token);
  if (cacheItem != cache_.end()) {
    return cacheItem->second;
  }

  std::vector<std::wstring> word;
  for (auto &c : token) {
    word.emplace_back(1, c);
  }

  auto pairs = getPairs(word);
  if (pairs.empty()) {
    return token;
  }

  while (true) {
    wstring_pair bigram;
    size_t minRank = SIZE_MAX;
    for (auto &pair : pairs) {
      auto pairIt = bpeRanks_.find(pair);
      if (pairIt != bpeRanks_.end()) {
        size_t rank = pairIt->second;
        if (rank < minRank) {
          minRank = rank;
          bigram = pair;
        }
      }
    }
    if (minRank >= bpeRanks_.size()) {
      break;
    }

    std::vector<std::wstring> newWord;
    size_t i = 0;
    while (i < word.size()) {
      bool foundFirst = false;
      size_t j = 0;
      for (j = i; j < word.size(); j++) {
        if (bigram.first == word[j]) {
          foundFirst = true;
          break;
        }
      }
      if (foundFirst) {
        for (size_t idx = i; idx < j; idx++) {
          newWord.push_back(word[idx]);
        }
        i = j;
      } else {
        for (size_t idx = i; idx < word.size(); idx++) {
          newWord.push_back(word[idx]);
        }
        break;
      }

      if (word[i] == bigram.first && i < word.size() - 1 && word[i + 1] == bigram.second) {
        newWord.push_back(bigram.first + bigram.second);
        i += 2;
      } else {
        newWord.push_back(word[i]);
        i += 1;
      }
    }

    word = newWord;
    if (word.size() == 1) {
      break;
    } else {
      pairs = getPairs(word);
    }
  }

  std::wstring wordStr = word[0];
  for (size_t idx = 1; idx < word.size(); idx++) {
    wordStr += L" " + word[idx];
  }
  cache_[token] = wordStr;
  return wordStr;
}

std::vector<int32_t> Encoder::encode(const std::string &text) {
  std::vector<int32_t> ret;
  re2::StringPiece input(text);
  std::string token;
  while (RE2::FindAndConsume(&input, gEncoderPat_, &token)) {
    std::wstring wToken;
    for (uint8_t b : token) {
      wToken.push_back(byteEncoder_[b]);
    }
    auto bpeTokens = StringUtils::split(bpe(wToken), L' ');
    for (auto &bpeToken : bpeTokens) {
      ret.push_back(encoder_[bpeToken]);
    }
  }
  return ret;
}

std::string Encoder::decode(const std::vector<int32_t> &tokens) {
  std::wstring text;
  for (int32_t idx : tokens) {
    text += decoder_[idx];
  }

  std::string ret;
  for (wchar_t c : text) {
    ret.push_back(char(byteDecoder_.at(c)));
  }

  return ret;
}

Encoder Encoder::getEncoder(const std::string &modelsDir) {
  FUNCTION_TIMED();
  // read "encoder.json"
  std::fstream encoderFile(modelsDir + FILE_SEP + GPT_ENCODER_JSON, std::ios::in);
  if (!encoderFile.is_open()) {
    LOGE("open file failed: %s", GPT_ENCODER_JSON);
    return {};
  }
  std::unordered_map<std::wstring, int32_t> encoderMap = loadEncoderMap(encoderFile);

  // read "vocab.bpe"
  std::fstream vocabFile(modelsDir + FILE_SEP + GPT_VOCAB_BPE, std::ios::in);
  if (!vocabFile.is_open()) {
    LOGE("open file failed: %s", GPT_VOCAB_BPE);
    return {};
  }
  std::vector<wstring_pair> bpeMerges = loadVocabBpe(vocabFile);
  return {encoderMap, bpeMerges};
}

std::vector<wstring_pair> Encoder::getPairs(const std::vector<std::wstring> &word) {
  std::vector<wstring_pair> pairs;
  if (word.size() > 1) {
    auto previous = word[0];
    for (size_t i = 1; i < word.size(); i++) {
      pairs.emplace_back(previous, word[i]);
      previous = word[i];
    }
  }

  return pairs;
}

std::unordered_map<int32_t, wchar_t> Encoder::bytesToUnicode() {
  std::unordered_map<int32_t, wchar_t> b2u;

  auto setOriginByte = [&](int32_t start, int32_t end) {
    for (int32_t i = start; i <= end; i++) {
      b2u.insert({i, wchar_t(i)});
    }
  };

  setOriginByte(L'!', L'~');
  setOriginByte(L'¡', L'¬');
  setOriginByte(L'®', L'ÿ');

  int32_t n = 0;
  for (int32_t i = 0; i < 256; i++) {
    if (b2u.find(i) == b2u.end()) {
      b2u.insert({i, wchar_t(256 + n)});
      n++;
    }
  }

  return b2u;
}

std::unordered_map<std::wstring, int32_t> Encoder::loadEncoderMap(std::istream &in) {
  const auto json = FileUtils::parseJson(in);
  if (json.is_null()) {
    LOGE("parse file failed: %s", GPT_ENCODER_JSON);
    return {};
  }
  std::unordered_map<std::wstring, int32_t> encoderMap;
  for (auto &kv : json.object_items()) {
    encoderMap.insert({StringUtils::utf82wstring(kv.first), kv.second.int_value()});
  }

  return encoderMap;
}

std::vector<wstring_pair> Encoder::loadVocabBpe(std::istream &in) {
  std::vector<wstring_pair> vocabBpe;
  std::string line;
  while (std::getline(in, line)) {
    // skip empty line or comments
    if (line.empty() || line[0] == '#') {
      continue;
    }
    size_t sep = line.find(' ');
    vocabBpe.emplace_back(
        StringUtils::utf82wstring(line.substr(0, sep)),
        StringUtils::utf82wstring(line.substr(sep + 1))
    );
  }

  return vocabBpe;
}

}
