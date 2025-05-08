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

void app_tokenizer() {
  LOGI("app_tokenizer()");

  std::ifstream file("assets/text/shakespeare.txt", std::ios::in | std::ios::binary | std::ios::ate);
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
  bool isOk = tokenizer.initWithConfigHF("assets/llama31/tokenizer.json", "assets/llama31/tokenizer_config.json");
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

  assert(ids.size() == batch);
  assert(ids.front().size() == 1484267);

  auto timeCost = timer.elapseMillis();
  auto speed = (float)(content.size() * batch) / (float)timeCost * 1000.f / (1024 * 1024);
  LOGI("encode bytes: %lld, cost: %lld ms, %.1f MB / s", content.size() * batch, timeCost, speed);
}
