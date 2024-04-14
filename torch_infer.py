from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from moe import MNIST_MoE
import torch.nn.functional as F
from config import *

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=MNIST_MoE(INPUT_SIZE,EXPERTS,TOP,EMB_SIZE).to(DEVICE) # 模型
model.load_state_dict(torch.load('model.pth'))

model.eval()    # 预测模式

'''
对图片分类
'''
image,label=dataset[999]
print('正确分类:',label)
plt.imshow(image.permute(1,2,0))
plt.show()

logits=model(image.unsqueeze(0).to(DEVICE))
print('预测分类:',logits.argmax(-1).item())