# TinyGPT
TinyGPT is a minimal C++11 implementation of GPT-2 inference, built from scratch and mainly inspired by the [picoGPT](https://github.com/jaymody/picoGPT) project.

For more details, check out the accompanying blog post: [Write a GPT from scratch (TinyGPT)](https://robot9.me/write-gpt-from-scratch/)

## Features

- Fast BPE tokenizer, inspired by [tiktoken](https://github.com/openai/tiktoken).
- CPU and CUDA inference.
- KV cache enabled.

## Build and Run

### 1. Get the code

```bash
git clone --recurse-submodules https://github.com/keith2018/TinyGPT.git
```

### 2. Download GPT-2 model file
    
```python
python3 tools/download_gpt2_model.py
```
if success, you'll see the file `model_file.data` in directory `assets/gpt2`

### 3. Build and Run

```bash
mkdir build
cmake -B ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build --config Release
```

This will generate the executable file and copy assets to directory `app/bin`, then you can run the demo:

```bash
cd app/bin
./TinyGPT_demo
[DEBUG] TIMER TinyGPT::Model::loadModelGPT2: cost: 800 ms
[DEBUG] TIMER TinyGPT::Encoder::getEncoder: cost: 191 ms
INPUT:Alan Turing theorized that computers would one day become
GPT:the most powerful machines on the planet.
INPUT:exit
```

## Dependencies

- Tensor
  - `TinyTorch` [https://github.com/keith2018/TinyTorch](https://github.com/keith2018/TinyTorch)
- Json parser
  - `RapidJSON` [https://github.com/Tencent/rapidjson](https://github.com/Tencent/rapidjson)
- Regex (Tokenizer)
  - `RE2` [https://github.com/google/re2](https://github.com/google/re2)
  - `Abseil` [https://github.com/abseil/abseil-cpp](https://github.com/abseil/abseil-cpp)
- HashMap
  - `ankerl::unordered_dense` [https://github.com/martinus/unordered_dense](https://github.com/martinus/unordered_dense)

## License

This code is licensed under the MIT License (see [LICENSE](LICENSE)).
