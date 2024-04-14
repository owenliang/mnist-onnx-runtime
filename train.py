import torch 
from dataset import MNIST
from moe import MNIST_MoE
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 
from config import *

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=MNIST_MoE(INPUT_SIZE,EXPERTS,TOP,EMB_SIZE).to(DEVICE) # 模型

try:    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

'''
    训练模型
'''

EPOCH=50
BATCH_SIZE=64   # 从batch内选出10个不一样的数字

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)    # 数据加载器

iter_count=0
for epoch in range(EPOCH):
    for imgs,labels in dataloader:
        logits=model(imgs.to(DEVICE))
        
        loss=F.cross_entropy(logits,labels.to(DEVICE))
        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if iter_count%1000==0:
            print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
            torch.save(model.state_dict(),'.model.pth')
            os.replace('.model.pth','model.pth')
        iter_count+=1