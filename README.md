# TinyGPT
TinyGPT is a minimal C++ implementation of GPT-2 inference. This project was built from scratch primarily for educational purposes.

## Features

- Fast BPE tokenizer, inspired by [tiktoken](https://github.com/openai/tiktoken).
- CPU and CUDA inference.

`tinygpt::tokenizer` is faster than both [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) and [OpenAI tiktoken](https://github.com/openai/tiktoken)，the encoding speed was measured using the [~/benches/tokenizer.py](https://github.com/keith2018/TinyGPT/blob/main/benches/tokenizer.py) script on a machine with an Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz.

![](docs/bench.png)

## Build and Run

### 1. Get the code

```bash
git clone --recurse-submodules https://github.com/keith2018/TinyGPT.git
```

### 2. Download GPT-2 model files
    
```bash
git clone https://huggingface.co/openai-community/gpt2
```
if success, chang the path in file [`./demo/demo_gpt2.cpp`](https://github.com/keith2018/TinyGPT/blob/main/demo/demo_gpt2.cpp)

```cpp
const std::string GPT2_MODEL_DIR = "path to gpt2 model files";
```

### 3. Build and Run

```bash
mkdir build
cmake -B ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build --config Release
```

This will generate the executable file and copy assets to directory `demo/bin`, then you can run the demo:

```bash
cd demo/bin
./TinyGPT_demo
```

## Python binding

```python
# pip install .

import tinygpt
enc = tinygpt.Tokenizer()
enc.init_with_config_hf("tokenizer.json", "tokenizer_config.json")
ids = enc.encode("This is a test")
```

## Dependencies

- Tensor
  - `TinyTorch` [https://github.com/keith2018/TinyTorch](https://github.com/keith2018/TinyTorch)
- JsonParser
  - `RapidJSON` [https://github.com/Tencent/rapidjson](https://github.com/Tencent/rapidjson)
- Regex
  - `pcre2` [https://github.com/PCRE2Project/pcre2](https://github.com/PCRE2Project/pcre2)
- HashMap
  - `ankerl::unordered_dense` [https://github.com/martinus/unordered_dense](https://github.com/martinus/unordered_dense)
- ConcurrentQueue
  - `moodycamel::ConcurrentQueue` [https://github.com/cameron314/concurrentqueue](https://github.com/cameron314/concurrentqueue)

## License

This code is licensed under the MIT License (see [LICENSE](LICENSE)).
