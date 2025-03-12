import torch.nn as nn
import torch

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        
        self.gate = nn.Linear(input_dim, num_experts)

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, input_dim = x.size()
        device = x.device

        logits = self.gate(x)
        prob = torch.softmax(logits, dim=-1)
        top_k_prob, top_k_indices = torch.topk(prob, self.top_k, dim=-1)
        #print(f'top_k_prob : {top_k_prob}')
        #print(f'top_k_indices : {top_k_indices}')
        # 辅助损失函数
        if self.training:
            importance = prob.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)

            # load balance loss
            mask = torch.zeros_like(prob, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)
            routing_probs = prob * mask
            expert_usage = mask.float().mean(0)
            routing_weights = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()
            aux_loss = importance_loss + load_balance_loss  
        else:
            aux_loss = 0

        # 专家分配
        flat_indices = top_k_indices.view(-1)
        flat_prob = top_k_prob.view(-1)
        sample_indices = torch.arange(batch_size, device=device)[:, None]
        sample_indices = sample_indices.expand(-1, self.top_k).flatten()
        
        # 输出初始化
        outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features, device=device)
        # 处理每个专家， 决定哪些X送入到哪些专家中
        # 每一个batch都会被送进top-k个专家中，因为维度为[batch, top-k]
        for expert_idx in range(self.num_experts):
            # 1. 遍历128个经过softmax 筛选的最大值indices,他的值范围为0~7
            # 2. 如果第7和第99个indices被选中，其值为0， 即选中了第0个专家
            # 2. 找到第7和99是在哪个batch中
            # 3. 在第7, 99个 index 中, 并且获取其概率，即第7, 99个的概率
            # 4. 获取第3, 49 batch的tensor
            # 5. 将第3, 49batch的tensor送入到专家0中
            # 6. 并且因为同一个batch可能有多个不同的专家index， 所以同一个batch将会送进不同的专家中
            expert_mask = flat_indices == expert_idx
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_prob[expert_mask]

            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]
            if len(expert_samples) == 0:
                continue    
            for k in expert_samples:
                if k == 3:
                    print(f"第三个batch 将会送进{expert_idx}专家中")
            expert_input = x[expert_samples]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            outputs.index_add_(0, expert_samples, weighted_output)
        return outputs, aux_loss


if __name__ == "__main__":
    input_dim = 128
    output_dim = 256
    hidden_dim = 512
    num_experts = 8
    top_k = 2
    expert_capacity = 32
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)

    for _ in range(100):
        x= torch.rand(batch_size, input_dim).to(device)
        moe.train()
        output, loss = moe(x)
        print(f'Training output shape: {output.shape}')
        print(f'Training Auxiliary loss: {loss.item():.4f}')
    print("=" * 80)
    moe.eval()
    output, _ = moe(x)
    print(f'output shape: {output.shape}')