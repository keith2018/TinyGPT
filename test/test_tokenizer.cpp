/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"
#include "tokenizer/Tokenizer.h"

using namespace tinygpt;
using Behavior = tokenizer::SplitDelimiterBehavior;

TEST(TEST_tokenizer, pretokenize_split_removed) {
  auto text = "Hello,,, world! This is a test.";

  auto split = std::make_shared<tokenizer::Split>(",", Behavior::REMOVED);
  auto actual = split->preTokenize(text);
  std::vector<std::string> expected = {"Hello", " world! This is a test."};
  EXPECT_EQ(expected, getStrings(actual));
}

TEST(TEST_tokenizer, pretokenize_split_isolated) {
  auto text = "Hello,,, world! This is a test.";

  auto split = std::make_shared<tokenizer::Split>(",", Behavior::ISOLATED);
  auto actual = split->preTokenize(text);
  std::vector<std::string> expected = {"Hello", ",", ",", ",", " world! This is a test."};
  EXPECT_EQ(expected, getStrings(actual));
}

TEST(TEST_tokenizer, pretokenize_split_merged_with_previous) {
  auto text = "Hello,,, world! This is a test.";

  auto split = std::make_shared<tokenizer::Split>(",", Behavior::MERGED_WITH_PREVIOUS);
  auto actual = split->preTokenize(text);
  std::vector<std::string> expected = {"Hello,", ",", ",", " world! This is a test."};
  EXPECT_EQ(expected, getStrings(actual));
}

TEST(TEST_tokenizer, pretokenize_split_merged_with_next) {
  auto text = "Hello,,, world! This is a test.";

  auto split = std::make_shared<tokenizer::Split>(",", Behavior::MERGED_WITH_NEXT);
  auto actual = split->preTokenize(text);
  std::vector<std::string> expected = {"Hello", ",", ",", ", world! This is a test."};
  EXPECT_EQ(expected, getStrings(actual));
}

TEST(TEST_tokenizer, pretokenize_split_contiguous) {
  auto text = "Hello,,, world! This is a test.";

  auto split = std::make_shared<tokenizer::Split>(",", Behavior::CONTIGUOUS);
  auto actual = split->preTokenize(text);
  std::vector<std::string> expected = {"Hello", ",,,", " world! This is a test."};
  EXPECT_EQ(expected, getStrings(actual));
}

TEST(TEST_tokenizer, pretokenize_bytelevel) {
  auto text = "Hello,,, world! 你好 ";

  auto byteLevel = std::make_shared<tokenizer::ByteLevel>(true, true);
  auto actual = byteLevel->preTokenize(text);
  std::vector<std::string> expected = {"ĠHello", ",,,", "Ġworld", "!", "Ġä½łå¥½", "Ġ"};
  EXPECT_EQ(expected, getStrings(actual));

  byteLevel = std::make_shared<tokenizer::ByteLevel>(false, true);
  actual = byteLevel->preTokenize(text);
  expected = {"Hello", ",,,", "Ġworld", "!", "Ġä½łå¥½", "Ġ"};
  EXPECT_EQ(expected, getStrings(actual));

  byteLevel = std::make_shared<tokenizer::ByteLevel>(true, false);
  actual = byteLevel->preTokenize(text);
  expected = {"ĠHello,,,Ġworld!Ġä½łå¥½Ġ"};
  EXPECT_EQ(expected, getStrings(actual));

  byteLevel = std::make_shared<tokenizer::ByteLevel>(false, false);
  actual = byteLevel->preTokenize(text);
  expected = {"Hello,,,Ġworld!Ġä½łå¥½Ġ"};
  EXPECT_EQ(expected, getStrings(actual));
}

inline bool loadTokenizer(tokenizer::Tokenizer &tokenizer, const std::string &dir) {
  return tokenizer.initWithConfig(dir + "/tokenizer.json", dir + "/tokenizer_config.json");
}

TEST(TEST_tokenizer, tokenizer_llama_31_8b) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Llama-3.1-8B");
  EXPECT_TRUE(initOk);

  std::map<std::string, std::vector<int32_t>> tokenPair = {
      {"hello world!", {128000, 15339, 1917, 0}},
      {"hello world!   ", {128000, 15339, 1917, 0, 262}},
      {"<｜User｜>Thanks for putting me into the right direction",
       {128000, 27, 104926, 1502, 104926, 29, 12947, 369, 10917, 757, 1139, 279, 1314, 5216}},
      {"hello，你好啊, thanks", {128000, 15339, 104660, 53901, 102856, 11, 9523}},
      {" ありがとうございます。 Arigatoo gozaimasu",
       {128000, 220, 97135, 61689, 1811, 1676, 343, 266, 2689, 733, 89, 2706, 96377}},
      {"你好😀🐶", {128000, 57668, 53901, 76460, 222, 9468, 238, 114}},
      {"   hello world!    ", {128000, 256, 24748, 1917, 0, 257}},
  };

  for (auto &[text, ids] : tokenPair) {
    auto encodeRet = tokenizer.encode(text);
    EXPECT_TRUE(encodeRet == ids);
    auto decodeRet = tokenizer.decode(ids);
    EXPECT_TRUE(decodeRet == tokenizer.bosTokenStr() + text);
  }
}

TEST(TEST_tokenizer, tokenizer_ds_r1_8b) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/DeepSeek-R1-Distill-Llama-8B");
  EXPECT_TRUE(initOk);

  std::map<std::string, std::vector<int32_t>> tokenPair = {
      {"hello world!", {128000, 15339, 1917, 0}},
      {"hello world!   ", {128000, 15339, 1917, 0, 262}},
      {"<｜User｜>Thanks for putting me into the right direction",
       {128000, 128011, 12947, 369, 10917, 757, 1139, 279, 1314, 5216}},
      {"hello，你好啊, thanks", {128000, 15339, 104660, 53901, 102856, 11, 9523}},
      {" ありがとうございます。 Arigatoo gozaimasu",
       {128000, 220, 97135, 61689, 1811, 1676, 343, 266, 2689, 733, 89, 2706, 96377}},
      {"你好😀🐶", {128000, 57668, 53901, 76460, 222, 9468, 238, 114}},
      {"   hello world!    ", {128000, 256, 24748, 1917, 0, 257}},
  };

  for (auto &[text, ids] : tokenPair) {
    auto encodeRet = tokenizer.encode(text);
    EXPECT_TRUE(encodeRet == ids);
    auto decodeRet = tokenizer.decode(ids);
    EXPECT_TRUE(decodeRet == tokenizer.bosTokenStr() + text);
  }
}

TEST(TEST_tokenizer, tokenizer_gpt2) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/gpt2");
  EXPECT_TRUE(initOk);

  std::map<std::string, std::vector<int32_t>> tokenPair = {
      {"hello world!", {31373, 995, 0}},
      {"Thanks for putting me into the right direction", {9690, 329, 5137, 502, 656, 262, 826, 4571}},
      {"hello，你好啊, thanks", {31373, 171, 120, 234, 19526, 254, 25001, 121, 161, 243, 232, 11, 5176}},
      {" ありがとうございます。 Arigatoo gozaimasu",
       {23294, 224,   28255, 35585, 30201, 29557, 2515, 242, 2515, 244,  18566,
        30159, 33623, 16764, 943,   328,   265,   2238, 467, 89,   1385, 27345}},
  };

  for (auto &[text, ids] : tokenPair) {
    auto encodeRet = tokenizer.encode(text);
    EXPECT_TRUE(encodeRet == ids);
    auto decodeRet = tokenizer.decode(ids);
    EXPECT_TRUE(decodeRet == text);
  }
}

TEST(TEST_tokenizer, tokenizer_qwen2) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen2.5-3B");
  EXPECT_TRUE(initOk);

  std::map<std::string, std::vector<int32_t>> tokenPair = {
      {"hello, world!", {14990, 11, 1879, 0}},
      {"hello world!   ", {14990, 1879, 0, 262}},
      {"<｜User｜>Thanks for putting me into the right direction",
       {27, 130957, 1474, 130957, 29, 12658, 369, 10687, 752, 1119, 279, 1290, 5106}},
      {"hello，你好啊, thanks", {14990, 3837, 108386, 103924, 11, 9339}},
      {" ありがとうございます。 Arigatoo gozaimasu", {220, 140052, 1773, 1644, 343, 266, 2624, 728, 89, 2640, 95277}},
      {"你好😀🐶", {108386, 141334, 145549}},
      {"   hello world!    ", {256, 23811, 1879, 0, 257}},
      {"Aujourd'hui, j'ai bu un café très fort.",
       {32, 9635, 76392, 87153, 11, 502, 33055, 1031, 650, 51950, 24901, 11845, 13}},
  };

  for (auto &[text, ids] : tokenPair) {
    auto encodeRet = tokenizer.encode(text);
    EXPECT_TRUE(encodeRet == ids);
    auto decodeRet = tokenizer.decode(ids);
    EXPECT_TRUE(decodeRet == text);
  }
}

TEST(TEST_tokenizer, tokenizer_batch) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/DeepSeek-R1-Distill-Llama-8B");
  EXPECT_TRUE(initOk);

  auto text = "hello world!";
  auto decodeText = std::string("<｜begin▁of▁sentence｜>") + text;
  std::vector<int32_t> ids = {128000, 15339, 1917, 0};

  auto ret = tokenizer.encodeBatch({text, text, text});
  EXPECT_TRUE(ret == std::vector({ids, ids, ids}));

  auto decodeRet = tokenizer.decodeBatch({ids, ids, ids});
  EXPECT_TRUE(decodeRet == std::vector({decodeText, decodeText, decodeText}));
}

TEST(TEST_tokenizer, tokenizer_long_text) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Llama-3.1-8B");
  EXPECT_TRUE(initOk);

  std::string text(500000, 'a');
  auto ids = tokenizer.encode(text);
  EXPECT_TRUE(ids.size() == 62501);
  EXPECT_TRUE(ids[0] == 128000);
  for (auto i = 1; i < ids.size(); i++) {
    EXPECT_TRUE(ids[i] == 70540);
  }
}
