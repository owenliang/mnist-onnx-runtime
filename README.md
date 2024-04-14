# mnist-onnx-runtime

MoE model for MNIST

Inference with ONNX Runtime with tensorRT backend

## 依赖

```
pip3 install torch torchvision torchaudio matplotlib numpy netron onnxruntime -i https://mirrors.aliyun.com/pypi/simple/
```

## 训练

```
python train.py
```

## torch原生推理

```
python torch_infer.py
正确率:98.93 耗时: 16.553289651870728 s
```

## onnxruntime cuda推理

```
python onnx_cuda_infer.py
正确率:98.93 耗时: 13.57606029510498 s
```

## onnxruntime tensorrt推理

```
```