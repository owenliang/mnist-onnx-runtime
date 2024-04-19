# mnist-onnx-runtime

基于MoE架构的MNIST分类模型，利用ONNX RUNTIME进行推理加速

MoE model for MNIST

Inference with ONNX Runtime with CPU/CUDA/TensorRT backend

## MoE模型原理

![](moe.png)

对于待计算的输入向量emb，

* 准备n个expert子网络
* 准备gateway linear层，输入emb可以输出n个概率值
* 选择n个概率中最高的top个位置，作为被选中的expert子网络
* emb分别被top个expert子网络运算，输出的emb分别与各自的概率值相乘，并pointwise相加，得到最终的输出向量emb

## ONNX RUNTIME

微软开源的通用推理引擎，基于ONNX模型存储格式+RUNTIME框架，对深度学习框架和底层硬件之间实现了解耦和抽象。

利用ONNX解决方案，可以对接主流深度学习框架训练的Model、自动选择并运行于多种硬件设备（CPU,GPU）、并能透明的基于tensorRT这样的英伟达加速技术进一步加速。

ONNXRuntime的API非常简单，支持Python、C++、Java等主流语言，因此可以很方便利用JAVA/C++这样的语言实现自己的多线程推理服务，即具备生产级应用的开发能力。

参考链接：

```
https://onnxruntime.ai/docs/get-started/with-python.html
```

## 关于torch的trace和script模式

MoE模型由于存在gateway和expert动态控制流，因此导出ONNX计算图时需要采用torch的script模式，而非静态图trace模式。

关于torch模型的trace模式和script模式差异，见refs/torch-jit.jpg

## 依赖

cuda11.8

```
https://developer.nvidia.com/cuda-11-8-0-download-archive
```

cuDNN 8.9
```
https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-860/install-guide/index.html
https://developer.nvidia.com/rdp/cudnn-archive
```

tensorrt8.6.0
```
wget 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz'
tar -xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/TensorRT-8.6.1.6/lib
pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
pip install TensorRT-8.6.1.6/graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl 
pip install TensorRT-8.6.1.6/onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl 
```

pip dependencies
```
pip3 install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/  --index-url https://download.pytorch.org/whl/cu118
pip3 install matplotlib numpy netron tensorrt==8.6.1 onnx onnxruntime onnxruntime-gpu tensorrt -i https://mirrors.aliyun.com/pypi/simple/ 
```

## 配置

config.py控制模型结构：

```
INPUT_SIZE=28*28
EXPERTS=8
TOP=2
EMB_SIZE=16
```

* EXPERTS：专家网络总数量
* TOP：实际参与计算的网络数量

## 训练

```
python train.py
```

## torch原生推理

```
python torch_infer.py
正确率:99.19 耗时: 17.579143285751343 s
```

## onnxruntime cuda推理

```
python onnx_cuda_infer.py
正确率:99.19 耗时: 13.025182962417603 s
```

## onnxruntime tensorrt推理

```
python onnx_trt_infer.py
正确率:99.19 耗时: 29.297020435333252 s  【TODO:性能不如CUDA，需要找原因】
```

## 关于Netron

ONNX官方推荐的模型查看工具，可以查看模型结构，以及查看模型参数