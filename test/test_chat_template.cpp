/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"
#include "tokenizer/ChatTemplate.h"
#include "tokenizer/Tokenizer.h"

using namespace tinygpt;

// --- Basic variable output ---
TEST(TEST_chat_template, basic_variable) {
  auto result = tokenizer::applyChatTemplate("Hello {{ name }}!", {}, true, "", "");
  // name is not set, so it outputs nothing (none → "")
  EXPECT_EQ(result, "Hello !");
}

TEST(TEST_chat_template, basic_text_only) {
  auto result = tokenizer::applyChatTemplate("Hello world!", {}, true, "", "");
  EXPECT_EQ(result, "Hello world!");
}

// --- String literal ---
TEST(TEST_chat_template, string_literal) {
  auto result = tokenizer::applyChatTemplate("{{ 'hello' }}", {}, false, "", "");
  EXPECT_EQ(result, "hello");
}

// --- Boolean and special variables ---
TEST(TEST_chat_template, builtin_variables) {
  auto result = tokenizer::applyChatTemplate("{{ bos_token }}{{ eos_token }}", {}, false, "<s>", "</s>");
  EXPECT_EQ(result, "<s></s>");
}

TEST(TEST_chat_template, add_generation_prompt_true) {
  auto result = tokenizer::applyChatTemplate("{% if add_generation_prompt %}GEN{% endif %}", {}, true, "", "");
  EXPECT_EQ(result, "GEN");
}

TEST(TEST_chat_template, add_generation_prompt_false) {
  auto result = tokenizer::applyChatTemplate("{% if add_generation_prompt %}GEN{% endif %}", {}, false, "", "");
  EXPECT_EQ(result, "");
}

// --- For loop with messages ---
TEST(TEST_chat_template, for_loop_messages) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "Hi"},
      {"assistant", "Hello"},
  };
  auto result =
      tokenizer::applyChatTemplate("{% for msg in messages %}[{{ msg.role }}:{{ msg.content }}]{% endfor %}", messages);
  EXPECT_EQ(result, "[user:Hi][assistant:Hello]");
}

TEST(TEST_chat_template, for_loop_index_access) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "Hi"},
  };
  auto result = tokenizer::applyChatTemplate(
      "{% for msg in messages %}{{ msg['role'] }}:{{ msg['content'] }}{% endfor %}", messages);
  EXPECT_EQ(result, "user:Hi");
}

// --- Loop variables ---
TEST(TEST_chat_template, loop_first_last) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "A"},
      {"user", "B"},
      {"user", "C"},
  };
  auto result = tokenizer::applyChatTemplate(
      "{% for msg in messages %}"
      "{% if loop.first %}FIRST{% endif %}"
      "{% if loop.last %}LAST{% endif %}"
      "{{ msg.content }}"
      "{% endfor %}",
      messages);
  EXPECT_EQ(result, "FIRSTABLASTC");
}

TEST(TEST_chat_template, loop_index) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "A"},
      {"user", "B"},
  };
  auto result = tokenizer::applyChatTemplate("{% for msg in messages %}{{ loop.index0 }}{% endfor %}", messages);
  EXPECT_EQ(result, "01");
}

// --- If / elif / else ---
TEST(TEST_chat_template, if_elif_else) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "sys"},
      {"user", "usr"},
      {"assistant", "ast"},
  };
  auto result = tokenizer::applyChatTemplate(
      "{% for msg in messages %}"
      "{% if msg.role == 'system' %}S"
      "{% elif msg.role == 'user' %}U"
      "{% else %}A"
      "{% endif %}"
      "{% endfor %}",
      messages);
  EXPECT_EQ(result, "SUA");
}

// --- String comparison with != ---
TEST(TEST_chat_template, not_equal) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "hi"},
  };
  auto result = tokenizer::applyChatTemplate(
      "{% for msg in messages %}{% if msg.role != 'system' %}OK{% endif %}{% endfor %}", messages);
  EXPECT_EQ(result, "OK");
}

// --- Boolean operators: and / or / not ---
TEST(TEST_chat_template, bool_and) {
  auto result = tokenizer::applyChatTemplate("{% if true and true %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

TEST(TEST_chat_template, bool_or) {
  auto result = tokenizer::applyChatTemplate("{% if false or true %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

TEST(TEST_chat_template, bool_not) {
  auto result = tokenizer::applyChatTemplate("{% if not false %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

// --- Whitespace control ---
TEST(TEST_chat_template, trim_left) {
  auto result = tokenizer::applyChatTemplate("hello   {%- if true %} world{% endif %}", {});
  EXPECT_EQ(result, "hello world");
}

TEST(TEST_chat_template, trim_right) {
  auto result = tokenizer::applyChatTemplate("{% if true -%}   hello{% endif %}", {});
  EXPECT_EQ(result, "hello");
}

TEST(TEST_chat_template, trim_both) {
  auto result = tokenizer::applyChatTemplate("A  {%- if true -%}  B  {%- endif -%}  C", {});
  EXPECT_EQ(result, "ABC");
}

TEST(TEST_chat_template, trim_var) {
  auto result = tokenizer::applyChatTemplate("hello   {{- ' world' }}", {});
  EXPECT_EQ(result, "hello world");
}

// --- Filter: trim ---
TEST(TEST_chat_template, filter_trim) {
  auto result = tokenizer::applyChatTemplate("{{ '  hello  ' | trim }}", {});
  EXPECT_EQ(result, "hello");
}

// --- Filter: length ---
TEST(TEST_chat_template, filter_length) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "A"},
      {"user", "B"},
      {"user", "C"},
  };
  auto result = tokenizer::applyChatTemplate("{{ messages | length }}", messages);
  EXPECT_EQ(result, "3");
}

// --- Filter: upper / lower ---
TEST(TEST_chat_template, filter_upper) {
  auto result = tokenizer::applyChatTemplate("{{ 'hello' | upper }}", {});
  EXPECT_EQ(result, "HELLO");
}

TEST(TEST_chat_template, filter_lower) {
  auto result = tokenizer::applyChatTemplate("{{ 'HELLO' | lower }}", {});
  EXPECT_EQ(result, "hello");
}

// --- Filter: default ---
TEST(TEST_chat_template, filter_default) {
  auto result = tokenizer::applyChatTemplate("{{ undefined_var | default('fallback') }}", {});
  EXPECT_EQ(result, "fallback");
}

// --- String concatenation with ~ ---
TEST(TEST_chat_template, tilde_concat) {
  auto result = tokenizer::applyChatTemplate("{{ 'hello' ~ ' ' ~ 'world' }}", {});
  EXPECT_EQ(result, "hello world");
}

// --- String concatenation with + ---
TEST(TEST_chat_template, plus_concat) {
  auto result = tokenizer::applyChatTemplate("{{ 'hello' + ' world' }}", {});
  EXPECT_EQ(result, "hello world");
}

// --- Set variable ---
TEST(TEST_chat_template, set_variable) {
  auto result = tokenizer::applyChatTemplate("{% set x = 'hello' %}{{ x }}", {});
  EXPECT_EQ(result, "hello");
}

// --- Integer operations ---
TEST(TEST_chat_template, int_modulo) {
  auto result = tokenizer::applyChatTemplate("{{ 5 % 2 }}", {});
  EXPECT_EQ(result, "1");
}

TEST(TEST_chat_template, int_comparison) {
  auto result = tokenizer::applyChatTemplate("{% if 3 > 2 %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

// --- is defined / is not defined ---
TEST(TEST_chat_template, is_defined) {
  auto result = tokenizer::applyChatTemplate("{% if bos_token is defined %}YES{% endif %}", {}, false, "<s>", "");
  EXPECT_EQ(result, "YES");
}

TEST(TEST_chat_template, is_not_defined) {
  auto result = tokenizer::applyChatTemplate("{% if unknown_var is not defined %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

// --- Escape sequences in strings ---
TEST(TEST_chat_template, escape_newline) {
  auto result = tokenizer::applyChatTemplate("{{ 'line1\\nline2' }}", {});
  EXPECT_EQ(result, "line1\nline2");
}

// --- Nested if inside for ---
TEST(TEST_chat_template, nested_if_in_for) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "Be helpful"},
      {"user", "Hello"},
  };
  auto result = tokenizer::applyChatTemplate(
      "{% for msg in messages %}"
      "{% if msg.role == 'system' %}"
      "[SYS]{{ msg.content }}[/SYS]"
      "{% elif msg.role == 'user' %}"
      "[USR]{{ msg.content }}[/USR]"
      "{% endif %}"
      "{% endfor %}",
      messages);
  EXPECT_EQ(result, "[SYS]Be helpful[/SYS][USR]Hello[/USR]");
}

// --- Realistic Llama3-style template ---
TEST(TEST_chat_template, llama3_style) {
  std::string tmpl =
      "{{ bos_token }}"
      "{% for message in messages %}"
      "{% if message['role'] == 'system' %}"
      "<|start_header_id|>system<|end_header_id|>\n\n"
      "{{ message['content'] }}<|eot_id|>"
      "{% elif message['role'] == 'user' %}"
      "<|start_header_id|>user<|end_header_id|>\n\n"
      "{{ message['content'] }}<|eot_id|>"
      "{% elif message['role'] == 'assistant' %}"
      "<|start_header_id|>assistant<|end_header_id|>\n\n"
      "{{ message['content'] }}<|eot_id|>"
      "{% endif %}"
      "{% endfor %}"
      "{% if add_generation_prompt %}"
      "<|start_header_id|>assistant<|end_header_id|>\n\n"
      "{% endif %}";

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "Hello!"},
  };

  auto result = tokenizer::applyChatTemplate(tmpl, messages, true, "<|begin_of_text|>", "<|eot_id|>");

  std::string expected =
      "<|begin_of_text|>"
      "<|start_header_id|>system<|end_header_id|>\n\n"
      "You are a helpful assistant.<|eot_id|>"
      "<|start_header_id|>user<|end_header_id|>\n\n"
      "Hello!<|eot_id|>"
      "<|start_header_id|>assistant<|end_header_id|>\n\n";

  EXPECT_EQ(result, expected);
}

// --- Realistic ChatML-style template ---
TEST(TEST_chat_template, chatml_style) {
  std::string tmpl =
      "{% for message in messages %}"
      "<|im_start|>{{ message.role }}\n"
      "{{ message.content }}<|im_end|>\n"
      "{% endfor %}"
      "{% if add_generation_prompt %}"
      "<|im_start|>assistant\n"
      "{% endif %}";

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "Hello!"},
  };

  auto result = tokenizer::applyChatTemplate(tmpl, messages, true, "", "");

  std::string expected =
      "<|im_start|>system\n"
      "You are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n"
      "Hello!<|im_end|>\n"
      "<|im_start|>assistant\n";

  EXPECT_EQ(result, expected);
}

// --- Whitespace control in realistic template ---
TEST(TEST_chat_template, whitespace_control_realistic) {
  std::string tmpl =
      "{%- for message in messages %}"
      "{%- if message.role == 'user' -%}"
      "User: {{ message.content }}\n"
      "{% elif message.role == 'assistant' -%}"
      "Assistant: {{ message.content }}\n"
      "{% endif -%}"
      "{%- endfor -%}"
      "{%- if add_generation_prompt -%}"
      "Assistant: "
      "{% endif -%}";

  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "Hello"},
      {"assistant", "Hi there"},
      {"user", "How are you?"},
  };

  auto result = tokenizer::applyChatTemplate(tmpl, messages, true, "", "");

  std::string expected =
      "User: Hello\n"
      "Assistant: Hi there\n"
      "User: How are you?\n"
      "Assistant: ";

  EXPECT_EQ(result, expected);
}

// --- Empty messages ---
TEST(TEST_chat_template, empty_messages) {
  auto result = tokenizer::applyChatTemplate("{% for msg in messages %}X{% endfor %}", {});
  EXPECT_EQ(result, "");
}

// --- Method call: strip ---
TEST(TEST_chat_template, method_strip) {
  auto result = tokenizer::applyChatTemplate("{{ '  hello  '.strip() }}", {});
  EXPECT_EQ(result, "hello");
}

// --- Namespace-like set in loop ---
TEST(TEST_chat_template, set_in_loop_scope) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "A"},
      {"user", "B"},
  };
  auto result = tokenizer::applyChatTemplate(
      "{% set count = 0 %}"
      "{% for msg in messages %}"
      "{% set count = loop.index %}"
      "{% endfor %}"
      "{{ count }}",
      messages);
  // count in outer scope remains 0 because set inside for-loop is in inner scope
  EXPECT_EQ(result, "0");
}

// --- in operator ---
TEST(TEST_chat_template, in_operator) {
  auto result = tokenizer::applyChatTemplate("{% if 'hello' in 'hello world' %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

TEST(TEST_chat_template, not_in_operator) {
  auto result = tokenizer::applyChatTemplate("{% if 'xyz' not in 'hello world' %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

// --- Filter: first / last ---
TEST(TEST_chat_template, filter_first) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "First"},
      {"user", "Last"},
  };
  auto result = tokenizer::applyChatTemplate("{{ messages | first }}", messages);
  EXPECT_FALSE(result.empty());
}

// --- is none / is not none ---
TEST(TEST_chat_template, is_none) {
  auto result = tokenizer::applyChatTemplate("{% if unknown is none %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

TEST(TEST_chat_template, is_not_none) {
  auto result = tokenizer::applyChatTemplate("{% if bos_token is not none %}YES{% endif %}", {}, false, "<s>", "");
  EXPECT_EQ(result, "YES");
}

// --- namespace() for mutable state across loop iterations ---
TEST(TEST_chat_template, namespace_basic) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "sys"},
      {"user", "usr"},
  };
  auto result = tokenizer::applyChatTemplate(
      "{% set ns = namespace(found=false) %}"
      "{% for msg in messages %}"
      "{% if msg.role == 'system' %}"
      "{% set ns.found = true %}"
      "{% endif %}"
      "{% endfor %}"
      "{% if ns.found %}FOUND{% endif %}",
      messages);
  EXPECT_EQ(result, "FOUND");
}

// --- String split method ---
TEST(TEST_chat_template, method_split) {
  auto result = tokenizer::applyChatTemplate("{{ 'a-b-c'.split('-')[1] }}", {});
  EXPECT_EQ(result, "b");
}

TEST(TEST_chat_template, method_split_negative_index) {
  auto result = tokenizer::applyChatTemplate("{{ 'hello</think>world'.split('</think>')[-1] }}", {});
  EXPECT_EQ(result, "world");
}

// --- Integer subtraction ---
TEST(TEST_chat_template, int_subtraction) {
  auto result = tokenizer::applyChatTemplate("{{ 5 - 3 }}", {});
  EXPECT_EQ(result, "2");
}

// --- Negative integer literal ---
TEST(TEST_chat_template, negative_index) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "First"},
      {"user", "Last"},
  };
  auto result = tokenizer::applyChatTemplate("{{ messages[-1].content }}", messages);
  EXPECT_EQ(result, "Last");
}

// --- Message list integer indexing ---
TEST(TEST_chat_template, message_list_index) {
  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "sys"},
      {"user", "usr"},
  };
  auto result = tokenizer::applyChatTemplate("{{ messages[0]['role'] }}", messages);
  EXPECT_EQ(result, "system");
}

// --- startswith / endswith ---
TEST(TEST_chat_template, method_startswith) {
  auto result = tokenizer::applyChatTemplate("{% if 'hello world'.startswith('hello') %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

TEST(TEST_chat_template, method_endswith) {
  auto result = tokenizer::applyChatTemplate("{% if 'hello world'.endswith('world') %}YES{% endif %}", {});
  EXPECT_EQ(result, "YES");
}

// --- lstrip / rstrip ---
TEST(TEST_chat_template, method_lstrip) {
  auto result = tokenizer::applyChatTemplate("{{ '\\nhello'.lstrip('\\n') }}", {});
  EXPECT_EQ(result, "hello");
}

TEST(TEST_chat_template, method_rstrip) {
  auto result = tokenizer::applyChatTemplate("{{ 'hello\\n'.rstrip('\\n') }}", {});
  EXPECT_EQ(result, "hello");
}

// --- Undefined attribute access returns none (falsy) ---
TEST(TEST_chat_template, undefined_attribute_falsy) {
  std::vector<tokenizer::ChatMessage> messages = {{"user", "hi"}};
  auto result = tokenizer::applyChatTemplate(
      "{% for msg in messages %}{% if msg.tool_calls %}HAS_TOOLS{% else %}NO_TOOLS{% endif %}{% endfor %}", messages);
  EXPECT_EQ(result, "NO_TOOLS");
}

// --- Undefined variable is falsy ---
TEST(TEST_chat_template, undefined_variable_falsy) {
  auto result = tokenizer::applyChatTemplate("{% if tools %}HAS_TOOLS{% else %}NO_TOOLS{% endif %}", {});
  EXPECT_EQ(result, "NO_TOOLS");
}

// =========================================================================
// Integration tests with real model tokenizer configs
// =========================================================================
inline bool loadTokenizer(tokenizer::Tokenizer& tokenizer, const std::string& dir) {
  return tokenizer.initWithConfig(dir + "/tokenizer.json", dir + "/tokenizer_config.json");
}

// --- DeepSeek-R1-Distill-Llama-8B: simple user message ---
TEST(TEST_chat_template, deepseek_r1_simple_user) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/DeepSeek-R1-Distill-Llama-8B");
  ASSERT_TRUE(initOk);
  ASSERT_TRUE(tokenizer.hasChatTemplate());

  std::vector<tokenizer::ChatMessage> messages = {{"user", "Hello!"}};
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string expected =
      "\xef\xbd\x9c"
      "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"  // <｜begin▁of▁sentence｜>
      "<\xef\xbd\x9c"
      "User\xef\xbd\x9c>"  // <｜User｜>
      "Hello!"
      "<\xef\xbd\x9c"
      "Assistant\xef\xbd\x9c>"  // <｜Assistant｜>
      "<think>\n";

  // use bosTokenStr to build expected string for clarity
  std::string bos = tokenizer.bosTokenStr();
  expected = bos +
             "<\xef\xbd\x9c"
             "User\xef\xbd\x9c>Hello!<\xef\xbd\x9c"
             "Assistant\xef\xbd\x9c><think>\n";

  EXPECT_EQ(result, expected);
}

// --- DeepSeek-R1-Distill-Llama-8B: with system message ---
TEST(TEST_chat_template, deepseek_r1_with_system) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/DeepSeek-R1-Distill-Llama-8B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "Hello!"},
  };
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string bos = tokenizer.bosTokenStr();
  std::string expected = bos +
                         "You are a helpful assistant."
                         "<\xef\xbd\x9c"
                         "User\xef\xbd\x9c>Hello!"
                         "<\xef\xbd\x9c"
                         "Assistant\xef\xbd\x9c><think>\n";

  EXPECT_EQ(result, expected);
}

// --- DeepSeek-R1-Distill-Llama-8B: multi-turn ---
TEST(TEST_chat_template, deepseek_r1_multi_turn) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/DeepSeek-R1-Distill-Llama-8B");
  ASSERT_TRUE(initOk);

  std::string eos = tokenizer.eosTokenStr();
  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "What is 1+1?"},
      {"assistant", "The answer is 2."},
      {"user", "Thanks!"},
  };
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string bos = tokenizer.bosTokenStr();
  std::string expected = bos +
                         "You are a helpful assistant."
                         "<\xef\xbd\x9c"
                         "User\xef\xbd\x9c>What is 1+1?"
                         "<\xef\xbd\x9c"
                         "Assistant\xef\xbd\x9c>The answer is 2." +
                         eos +
                         "<\xef\xbd\x9c"
                         "User\xef\xbd\x9c>Thanks!"
                         "<\xef\xbd\x9c"
                         "Assistant\xef\xbd\x9c><think>\n";

  EXPECT_EQ(result, expected);
}

// --- Qwen2.5-3B: simple user message ---
TEST(TEST_chat_template, qwen25_simple_user) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen2.5-3B");
  ASSERT_TRUE(initOk);
  ASSERT_TRUE(tokenizer.hasChatTemplate());

  std::vector<tokenizer::ChatMessage> messages = {{"user", "Hello!"}};
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello!<|im_end|>\n"
      "<|im_start|>assistant\n";

  EXPECT_EQ(result, expected);
}

// --- Qwen2.5-3B: with system message ---
TEST(TEST_chat_template, qwen25_with_system) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen2.5-3B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "Hello!"},
  };
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello!<|im_end|>\n"
      "<|im_start|>assistant\n";

  EXPECT_EQ(result, expected);
}

// --- Qwen2.5-3B: multi-turn ---
TEST(TEST_chat_template, qwen25_multi_turn) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen2.5-3B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "What is 1+1?"},
      {"assistant", "The answer is 2."},
      {"user", "Thanks!"},
  };
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nWhat is 1+1?<|im_end|>\n"
      "<|im_start|>assistant\nThe answer is 2.<|im_end|>\n"
      "<|im_start|>user\nThanks!<|im_end|>\n"
      "<|im_start|>assistant\n";

  EXPECT_EQ(result, expected);
}

// --- Qwen3-0.6B: simple user message ---
TEST(TEST_chat_template, qwen3_simple_user) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen3-0.6B");
  ASSERT_TRUE(initOk);
  ASSERT_TRUE(tokenizer.hasChatTemplate());

  std::vector<tokenizer::ChatMessage> messages = {{"user", "Hello!"}};
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string expected =
      "<|im_start|>user\nHello!<|im_end|>\n"
      "<|im_start|>assistant\n";

  EXPECT_EQ(result, expected);
}

// --- Qwen3-0.6B: with system message ---
TEST(TEST_chat_template, qwen3_with_system) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen3-0.6B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "Hello!"},
  };
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello!<|im_end|>\n"
      "<|im_start|>assistant\n";

  EXPECT_EQ(result, expected);
}

// --- Qwen3-0.6B: multi-turn ---
TEST(TEST_chat_template, qwen3_multi_turn) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen3-0.6B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "What is 1+1?"},
      {"assistant", "The answer is 2."},
      {"user", "Thanks!"},
  };
  auto result = tokenizer.applyChatTemplate(messages, true);

  std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nWhat is 1+1?<|im_end|>\n"
      "<|im_start|>assistant\nThe answer is 2.<|im_end|>\n"
      "<|im_start|>user\nThanks!<|im_end|>\n"
      "<|im_start|>assistant\n";

  EXPECT_EQ(result, expected);
}

// --- Qwen3-0.6B: no generation prompt ---
TEST(TEST_chat_template, qwen3_no_gen_prompt) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen3-0.6B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"system", "You are a helpful assistant."},
      {"user", "Hello!"},
  };
  auto result = tokenizer.applyChatTemplate(messages, false);

  std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello!<|im_end|>\n";

  EXPECT_EQ(result, expected);
}

// --- Verify tokenizer::applyChatTemplate + encode roundtrip ---
TEST(TEST_chat_template, qwen25_template_then_encode) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/Qwen2.5-3B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "Hi"},
  };
  auto text = tokenizer.applyChatTemplate(messages, true);
  EXPECT_FALSE(text.empty());

  auto ids = tokenizer.encode(text);
  EXPECT_FALSE(ids.empty());

  auto decoded = tokenizer.decode(ids);
  EXPECT_EQ(decoded, text);
}

TEST(TEST_chat_template, deepseek_r1_template_then_encode) {
  tokenizer::Tokenizer tokenizer;
  bool initOk = loadTokenizer(tokenizer, "assets/tokenizer/DeepSeek-R1-Distill-Llama-8B");
  ASSERT_TRUE(initOk);

  std::vector<tokenizer::ChatMessage> messages = {
      {"user", "Hi"},
  };
  auto text = tokenizer.applyChatTemplate(messages, true);
  EXPECT_FALSE(text.empty());

  auto ids = tokenizer.encode(text);
  EXPECT_FALSE(ids.empty());
}
