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
    def __init__(self,experts,top,emb_size):
        super().__init__()
        self.experts=nn.ModuleList([Expert(emb_size) for _ in range(experts)])
        self.top=top
        self.gate=nn.Linear(emb_size,experts)
        
    def forward(self,x):    # x: (batch,seq_len,emb)
        x_shape=x.shape
        
        x=x.reshape(-1,x_shape[-1]) # (batch*seq_len,emb)
        
        # gates 
        gate_logits=self.gate(x)    # (batch*seq_len,experts)
        gate_prob=softmax(gate_logits,dim=-1)   # (batch*seq_len,experts)
        
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
            y[top_index==expert_i]=y_expert
        
        # weighted sum experts
        top_weights=top_weights.view(-1,1).expand(-1,x.size(-1))  # (batch*seq_len*top,emb)
        y=y*top_weights
        y=y.view(-1,self.top,x.size(-1))    # (batch*seq_len,top,emb)
        y=y.sum(dim=1)  # (batch*seq_len,emb)
        return y.view(x_shape)

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
        y=self.moe(y)
        return self.cls(y)

if __name__=='__main__':
    moe=MoE(experts=8,top=2,emb_size=16)
    x=torch.rand((5,10,16))
    y=moe(x)
    print(y.shape)
    
    mnist_moe=MNIST_MoE(input_size=784,experts=8,top=2,emb_size=16)
    x=torch.rand((5,1,28,28))
    y=mnist_moe(x)
    print(y.shape)