### SMLP

import torch
import torch.nn as nn

# ==============================================================================
# 组件 1：单层 SMLP 基础块
# ==============================================================================
class SMLPBlock(nn.Module):
    def __init__(self, seq_len, feature_dim, dropout=0.1):
        super(SMLPBlock, self).__init__()
        
        # --- Time Mixing (时间维度混合) ---
        # 捕捉过去 96 小时内各个时间点的相互依赖关系
        self.time_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len)
        )
        self.norm1 = nn.BatchNorm1d(seq_len)
        
        # --- Feature Mixing (特征维度混合) ---
        # 捕捉在同一时刻，温度、湿度、晴空指数、时间编码等特征之间的非线性交叉
        self.feature_mixer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.norm2 = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        
        # 1. Time Mixing (需要转置以在 seq_len 维度上做 Linear)
        x_time = x.transpose(1, 2)  # (batch, feature_dim, seq_len)
        x_time = self.time_mixer(x_time)
        x_time = x_time.transpose(1, 2) # 转回 (batch, seq_len, feature_dim)
        x = self.norm1(x + x_time)      # 残差连接 + 归一化 (防止梯度消失)
        
        # 2. Feature Mixing (直接在 feature_dim 维度上做 Linear)
        x_feat = self.feature_mixer(x)
        x = self.norm2(x + x_feat)      # 残差连接 + 归一化
        
        return x

# ==============================================================================
# 组件 2：SMLP 完整预测模型
# ==============================================================================
class SMLPModel(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, feature_dim=12, num_blocks=3, dropout=0.1):
        """
        seq_len: 历史回溯窗口长度 (如 96小时)
        pred_len: 未来预测窗口长度 (如 24小时)
        feature_dim: 输入特征的总维度 (包含物理先验、时序编码等)
        num_blocks: SMLP 核心块的堆叠层数
        """
        super(SMLPModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. 堆叠的 SMLP 核心骨干网络 (Backbone)
        self.blocks = nn.ModuleList([
            SMLPBlock(seq_len, feature_dim, dropout) for _ in range(num_blocks)
        ])
        
        # 2. 预测头 (Prediction Head) - 将 96 小时的隐变量展平，映射到未来 24 小时
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(seq_len * feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, pred_len)  # 输出未来 24 小时的预测值
        )
        
        # 3. 物理极值约束门控 (可选：显式利用晴空指数理论最大值控制输出上限)
        # 假设 feature_dim 的第 0 列是真实 UVI/GHI，第 1 列是理论晴空辐射(Clear Sky)
        self.relu = nn.ReLU() # 确保预测值非负

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        
        # 1. 特征提取与深度混合
        hidden = x
        for block in self.blocks:
            hidden = block(hidden)
            
        # 2. 生成初步预测
        out = self.flatten(hidden)
        pred = self.head(out)  # shape: (batch_size, pred_len)
        
        # 3. 物理边界软性截断 (物理合理性保障)
        pred = self.relu(pred) # 紫外线指数绝不可能为负数
        
        return pred

# ==============================================================================
# 本地模型尺寸与前向传播测试
# ==============================================================================
if __name__ == "__main__":
    # 模拟一个 Batch 的输入数据
    # Batch Size = 32, 历史 96 小时, 12 个特征 (含气象、时序编码、晴空指数等)
    dummy_input = torch.randn(32, 96, 12)
    
    # 实例化模型
    model = SMLPModel(seq_len=96, pred_len=24, feature_dim=12, num_blocks=3)
    
    # 前向传播测试
    output = model(dummy_input)
    
    print(f"输入数据维度: {dummy_input.shape}")
    print(f"预测输出维度: {output.shape} (预期为 [32, 24])")
    
    # 计算模型参数量 (证明其"轻量化"优势)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SMLP 模型总参数量: {total_params / 1e6:.4f} M (百万)")

    ### TXMixer Autoformer TFT  DLinear Persistent 均有开源库