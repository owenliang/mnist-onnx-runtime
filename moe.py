from torch import nn 
from torch import softmax
import torch 

class Expert(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        
        self.seq=nn.Sequential(
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size),
        )

    def forward(self,x):
        return self.seq(x)

# Mixture of Experts
class MoE(nn.Module):
    def __init__(self,experts,top,emb_size,w_importance=0.01):
        super().__init__()
        self.experts=nn.ModuleList([Expert(emb_size) for _ in range(experts)])
        self.top=top
        self.gate=nn.Linear(emb_size,experts)
        self.noise=nn.Linear(emb_size,experts)  # 给gate输出概率加噪音用
        self.w_importance=w_importance  # expert均衡用途(for loss)
        
    def forward(self,x):    # x: (batch,seq_len,emb)
        x_shape=x.shape
        
        x=x.reshape(-1,x_shape[-1]) # (batch*seq_len,emb)
        
        # gates 
        gate_logits=self.gate(x)    # (batch*seq_len,experts)
        gate_prob=softmax(gate_logits,dim=-1)   # (batch*seq_len,experts)
        
        # 2024-05-05 Noisy Top-K Gating，优化expert倾斜问题
        if self.training: # 仅训练时添加噪音
            noise=torch.randn_like(gate_prob)*nn.functional.softplus(self.noise(x)) # https://arxiv.org/pdf/1701.06538 , StandardNormal()*Softplus((x*W_noise))
            gate_prob=gate_prob+noise
        
        # top expert
        top_weights,top_index=torch.topk(gate_prob,k=self.top,dim=-1)   # top_weights: (batch*seq_len,top), top_index: (batch*seq_len,top)
        top_weights=softmax(top_weights,dim=-1)
        
        top_weights=top_weights.view(-1)    # (batch*seq_len*top)
        top_index=top_index.view(-1)    # (batch*seq_len*top)
        
        x=x.unsqueeze(1).expand(x.size(0),self.top,x.size(-1)).reshape(-1,x.size(-1)) # (batch*seq_len*top,emb)
        y=torch.zeros_like(x)   # (batch*seq_len*top,emb)
        
        # run by per expert
        for expert_i,expert_model in enumerate(self.experts):
            x_expert=x[top_index==expert_i] # (...,emb)
            y_expert=expert_model(x_expert)   # (...,emb)
            
            add_index=(top_index==expert_i).nonzero().flatten() # 要修改的下标
            y=y.index_add(dim=0,index=add_index,source=y_expert)   # 等价于y[top_index==expert_i]=y_expert，为了保证计算图正确，保守用index_add算子
        
        # weighted sum experts
        top_weights=top_weights.view(-1,1).expand(-1,x.size(-1))  # (batch*seq_len*top,emb)
        y=y*top_weights
        y=y.view(-1,self.top,x.size(-1))    # (batch*seq_len,top,emb)
        y=y.sum(dim=1)  # (batch*seq_len,emb)
        
        # 2024-05-05 计算gate输出各expert的累计概率, 做一个loss让各累计概率尽量均衡，避免expert倾斜
        # https://arxiv.org/pdf/1701.06538 BALANCING EXPERT UTILIZATION
        if self.training:
            importance=gate_prob.sum(dim=0) # 将各expert打分各自求和 sum( (batch*seq_len,experts) , dim=0)
            # 求CV变异系数（也就是让expert们的概率差异变小）, CV=标准差/平均值
            importance_loss=self.w_importance*(torch.std(importance)/torch.mean(importance))**2
        else:
            importance_loss=None 
        return y.view(x_shape),gate_prob,importance_loss   # 2024-05-05 返回gate的输出用于debug其均衡效果, 返回均衡loss 

# MNIST分类
class MNIST_MoE(nn.Module):
    def __init__(self,input_size,experts,top,emb_size):
        super().__init__()
        self.emb=nn.Linear(input_size,emb_size)
        self.moe=MoE(experts,top,emb_size)
        self.cls=nn.Linear(emb_size,10)
        
    def forward(self,x):
        x=x.view(-1,784)
        y=self.emb(x)
        y,gate_prob,importance_loss=self.moe(y)
        return self.cls(y),gate_prob,importance_loss

if __name__=='__main__':
    moe=MoE(experts=8,top=2,emb_size=16)
    x=torch.rand((5,10,16))
    y,prob,imp_loss=moe(x)
    print(y.shape,prob.shape,imp_loss.shape)
    
    mnist_moe=MNIST_MoE(input_size=784,experts=8,top=2,emb_size=16)
    x=torch.rand((5,1,28,28))
    y,prob,imp_loss=mnist_moe(x)
    print(y.shape)