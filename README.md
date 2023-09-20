# test_tensorrt_cpp_load
for illustrated deserialized wrong in tensorrt


## ENVIROMENT
C++ use `TensorRT-8.6.1.6`\
In python pip list:
```python
graphsurgeon              0.4.6
numpy                     1.24.4
onnx                      1.14.1
onnx-graphsurgeon         0.3.12
onnxruntime-gpu           1.15.1
onnxsim                   0.4.33
tensorrt                  8.6.1
tensorrt-dispatch         8.6.1
tensorrt-lean             8.6.1
torch                     1.13.1+cu117
torchaudio                0.13.1+cu117
torchvision               0.14.1+cu117
```

## Installation
According to the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) to install

## In Python
The code to serialize mode and deserialize model. Finally the serialized model is generated in python code if you run like this:
```bash
$ python demo_trt.py
```
you can see the serialized model file `ResNet34_trackerOCR_36_450_20230627_half.engine`.

## In C++
I tried to deserialize in C++, but get some error:
```
Parsing model file!
Succeeded getting serialized engine!
INFO: Loaded engine size: 47 MiB
ERROR: 1: [dispatchStubs.cpp::deserializeEngine::14] Error Code 1: Internal Error (Unexpected call to stub)
Failed loading engine!
```

## Reproduce in C++
```bash
$ cp CMakeLists.txt_class CMakeLists.txt
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/your/cuda-11.7 -DTRT_ROOT_DIR=/path/to/your/TensorRT/dir/targets/x86_64-linux-gnu ..
$ make
$ cd ../bin
$ ./main_class
```

## write in a function && reproduce in C++
```bash
$ cp CMakeLists.txt_func CMakeLists.txt
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/your/cuda-11.7 -DTRT_ROOT_DIR=/path/to/your/TensorRT/dir/targets/x86_64-linux-gnu ..
$ make
$ cd ../bin
$ ./main_func
```
This operation is successed in deserialize model.

## Additional Instructions
In order to be able to verify whether the model checked in `trtexec` is normal, I commented out the serialization and deserialization operations of the extra information in the code. The model file generated by the current code is directly serialized and then saved in binary.


## Final Solution (Thank [@Data-Adventure](https://github.com/Data-Adventure))
[@Data-Adventure](https://github.com/Data-Adventure) and I firmly believe this is a dynamic linking bug when trying to generate dynamic libraries. When you try to generate a static library, there is no problem with static linking, and it can be compiled and executed successfully. I have also updated the corresponding cmake files, which can be reproduced by following the following code command:
```bash
$ cd /path/to/project_dir
$ cp CMakeLists.txt_class CMakeLists.txt
$ cp src/CMakeLists_static.txt src/CMakeLists.txt
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/your/cuda-11.7 -DTRT_ROOT_DIR=/path/to/your/TensorRT/dir/targets/x86_64-linux-gnu ..
$ make
$ cd ../bin
$ ./main_class
```
If you want reproduce the bug, you can follow the following code command:
```bash
$ cd /path/to/project_dir
$ cp CMakeLists.txt_class CMakeLists.txt
$ cp src/CMakeLists_shared.txt src/CMakeLists.txt
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/your/cuda-11.7 -DTRT_ROOT_DIR=/path/to/your/TensorRT/dir/targets/x86_64-linux-gnu ..
$ make
$ cd ../bin
$ ./main_class
```