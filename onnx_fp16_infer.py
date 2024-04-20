from torch import nn 
import torch 
import onnx 
import netron
import onnxruntime
from moe import MNIST_MoE
from config import * 
from dataset import MNIST
from torch.utils.data import DataLoader
import time 
from onnxruntime.quantization import quantize_dynamic
import os 
from onnxconverter_common import float16
 
EPOCH=10
BATCH_SIZE=64 

dataset=MNIST() # 数据集
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,num_workers=10,persistent_workers=True)    # 数据加载器

model=MNIST_MoE(INPUT_SIZE,EXPERTS,TOP,EMB_SIZE) # 模型
model.load_state_dict(torch.load('model.pth'))

model=torch.jit.script(model) # 带控制流的静态图导出

model.eval()    # 预测模式

# 导出onnx格式
torch.onnx.export(model,torch.rand((BATCH_SIZE,1,28,28)),f='model.onnx')

# 检查onnx导出正确
onnx_model=onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# fp16
model_fp16=float16.convert_float_to_float16(onnx_model)
onnx.save(model_fp16,'model-fp16.onnx')

# 推理
sess=onnxruntime.InferenceSession('model-fp16.onnx',providers=['CUDAExecutionProvider','CPUExecutionProvider'])

start_time=time.time()

correct=0
for epoch in range(EPOCH):
    for img,label in dataloader:
        batch_size=img.size(0)
        if img.size(0)!=BATCH_SIZE: # onnx输入尺寸固定，最后1个batch要补齐
            fills=torch.zeros(BATCH_SIZE-img.size(0),1,28,28)
            img=torch.concat((img,fills),dim=0)
        outputs=sess.run(output_names=None,input_feed={sess.get_inputs()[0].name:img.numpy().astype('float16')})  # 输入&输出
        logits=outputs[0][:batch_size]

        correct+=(logits.argmax(-1)==label.numpy()).sum()

print('正确率:%.2f'%(correct/(len(dataset)*EPOCH)*100),'耗时:',time.time()-start_time,'s')

# 展示onnx模型
netron.start('model-fp16.onnx')