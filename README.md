# TinyGPT
Tiny C++11 GPT-2 inference implementation from scratch, which is mainly based on the project [picoGPT](https://github.com/jaymody/picoGPT).

Accompanying blog post: [Write a GPT from scratch (TinyGPT)](https://robot9.me/write-gpt-from-scratch/)

## Core class

- [`Tensor`](src/Tensor.h): Tensor class similar to the [numpy](https://numpy.org/doc/1.25/reference/routines.html) interface.
- [`Model`](src/Model.h): GPT-2 model implementation with reference to [gpt2_pico.py](https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py).
- [`Tokenizer`](src/Tokenizer.h): BPE tokenizer with exactly the same logic as GPT-2 [encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py).


## Build and Run

### 1. Get the code

```bash
git clone --recurse-submodules https://github.com/keith2018/TinyGPT.git
```

### 2. Install Intel MKL(Math Kernel Library)

Official website: [Intel®-Optimized Math Library for Numerical Computing on CPUs & GPUs](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

### 3. Download GPT-2 model file
    
```python
python3 tools/download_gpt2_model.py
```
if success, you'll see the file `model_file.data` in directory `assets/gpt2`

### 4. Build and Run

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

- GEMM acceleration
  - `intel-mkl` [https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
- Json parser
  - `json11` [https://github.com/dropbox/json11](https://github.com/dropbox/json11)
- Tokenizer regular matching
  - `re2` [https://github.com/google/re2](https://github.com/google/re2)
  - `abseil-cpp` [https://github.com/abseil/abseil-cpp](https://github.com/abseil/abseil-cpp)

## License

This code is licensed under the MIT License (see [LICENSE](LICENSE)).
