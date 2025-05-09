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

TEST(TEST_tokenizer, tokenizer_llama31) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = tokenizer.initWithConfigHF("assets/llama31/tokenizer.json", "assets/llama31/tokenizer_config.json");
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
  };

  for (auto &[text, ids] : tokenPair) {
    auto encodeRet = tokenizer.encode(text);
    EXPECT_TRUE(encodeRet == ids);
    auto decodeRet = tokenizer.decode(ids);
    EXPECT_TRUE(decodeRet == tokenizer.bosTokenContent() + text);
  }
}

TEST(TEST_tokenizer, tokenizer_gpt2) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = tokenizer.initWithConfigGPT2("assets/gpt2/encoder.json", "assets/gpt2/vocab.bpe");
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

TEST(TEST_tokenizer, tokenizer_batch) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = tokenizer.initWithConfigHF("assets/llama31/tokenizer.json", "assets/llama31/tokenizer_config.json");
  EXPECT_TRUE(initOk);

  auto text = "hello world!";
  auto decodeText = std::string("<｜begin▁of▁sentence｜>") + text;
  std::vector<int32_t> ids = {128000, 15339, 1917, 0};

  auto ret = tokenizer.encodeBatch({text, text, text});
  EXPECT_TRUE(ret == std::vector({ids, ids, ids}));

  auto decodeRet = tokenizer.decodeBatch({ids, ids, ids});
  EXPECT_TRUE(decodeRet == std::vector({decodeText, decodeText, decodeText}));
}
