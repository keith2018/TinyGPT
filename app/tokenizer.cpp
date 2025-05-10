/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "tokenizer/Tokenizer.h"

#include <fstream>
#include <iostream>

#include "TinyTorch/Timer.h"

using namespace tinygpt;
using Behavior = tokenizer::SplitDelimiterBehavior;

constexpr char const* TEXT_PATH = "assets/text/shakespeare.txt";
constexpr char const* TOKENIZER_PATH = "assets/Llama-3.1-8B/tokenizer.json";
constexpr char const* TOKENIZER_CONFIG_PATH = "assets/Llama-3.1-8B/tokenizer_config.json";

void app_tokenizer() {
  LOGI("app_tokenizer()");

  std::ifstream file(TEXT_PATH, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file) {
    LOGE("Failed to open file.");
    return;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string content(size, '\0');
  if (!file.read(&content[0], size)) {
    LOGE("Failed to read file.");
    return;
  }

  tokenizer::Tokenizer tokenizer;
  bool isOk = tokenizer.initWithConfigHF(TOKENIZER_PATH, TOKENIZER_CONFIG_PATH);
  LOGI("Tokenizer init ok: %s", isOk ? "true" : "false");

  auto batch = 8;
#ifdef DEBUG
  batch = 1;
#endif

  std::vector<std::string> input;
  input.reserve(batch);
  for (auto i = 0; i < batch; i++) {
    input.emplace_back(content);
  }

  TinyTorch::Timer timer;
  timer.start();
  auto ids = tokenizer.encodeBatch(input, 4, false);
  timer.mark();

  auto timeCost = timer.elapseMillis();
  auto speed = (float)(content.size() * batch) / (float)timeCost * 1000.f / (1024 * 1024);
  LOGI("encode bytes: %lld, cost: %lld ms, %.1f MB / s", content.size() * batch, timeCost, speed);
}
