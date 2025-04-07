# TinyGPT
Tiny C++11 GPT-2 inference implementation from scratch, which is mainly based on the project [picoGPT](https://github.com/jaymody/picoGPT).

Accompanying blog post: [Write a GPT from scratch (TinyGPT)](https://robot9.me/write-gpt-from-scratch/)

## Core class

- [`Model`](src/Model.h): GPT-2 model implementation with reference to [gpt2_pico.py](https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py).
- [`Tokenizer`](src/Tokenizer.h): BPE tokenizer with exactly the same logic as GPT-2 [encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py).


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
  - `json11` [https://github.com/dropbox/json11](https://github.com/dropbox/json11)

## License

This code is licensed under the MIT License (see [LICENSE](LICENSE)).
