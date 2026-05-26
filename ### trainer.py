### trainer

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

# ==============================================================================
# 1. 时序滑动窗口数据集构建 (Time Series Dataset)
# ==============================================================================
class WeatherTimeSeriesDataset(Dataset):
    def __init__(self, data_matrix, target_matrix, seq_len=96, pred_len=24):
        """
        data_matrix: 包含所有特征的 numpy 数组 (样本数, 特征维度)
        target_matrix: 仅包含目标预测值 UVI 的 numpy 数组 (样本数, 1)
        seq_len: 历史观测长度 (默认 96 小时)
        pred_len: 未来预测长度 (默认 24 小时)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 确保可以通过滑动窗口截取到完整的 (x, y) 对
        self.valid_length = len(data_matrix) - seq_len - pred_len + 1
        self.x = torch.FloatTensor(data_matrix)
        self.y = torch.FloatTensor(target_matrix)

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        # 提取历史 96 小时的所有特征
        seq_x = self.x[idx : idx + self.seq_len, :]
        # 提取未来 24 小时的真实 UVI 值
        seq_y = self.y[idx + self.seq_len : idx + self.seq_len + self.pred_len, 0]
        return seq_x, seq_y

# ==============================================================================
# 2. 训练流程控制 (包含 Huber Loss, Warmup, 早停)
# ==============================================================================
def train_model(model, train_loader, val_loader, epochs=50, device='cuda'):
    model = model.to(device)
    
    # 使用 Huber Loss (Smooth L1 Loss) 替代普通 MSE，增强抗极端天气噪声的能力
    criterion = nn.SmoothL1Loss(beta=1.0) 
    
    # 采用 AdamW 优化器，设定 weight_decay 防止 MLP 过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 余弦退火学习率调度
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 早停机制 (Early Stopping) 配置
    patience = 7
    best_val_loss = float('inf')
    counter = 0
    best_model_weights = None
    
    print("🚀 开始模型训练...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # 计算损失 (确保输出维度和真实值对齐)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # 反向传播与优化
            loss.backward()
            
            # 梯度裁剪 (Gradient Clipping) 也是防梯度爆炸的利器
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # ====================
        # 验证集评估阶段
        # ====================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # ====================
        # 早停判断逻辑
        # ====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            counter = 0 # 重置耐心值
        else:
            counter += 1
            if counter >= patience:
                print(f"⚠️ 验证集损失在 {patience} 个 Epoch 内未下降，触发早停机制 (Early Stopping)！")
                break
                
    print(f"✅ 训练结束！最佳验证集损失: {best_val_loss:.4f}")
    
    # 恢复表现最好的那一轮的权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model

# ==============================================================================
# 3. 本地调试伪代码
# ==============================================================================
if __name__ == "__main__":
    # 假设特征维度 D = 12 (温度, 风速, kt, 各种正余弦时间编码等)
    dummy_data = np.random.randn(2000, 12)
    dummy_target = np.random.uniform(0, 15, (2000, 1))
    
    # 切分训练集与验证集 (时序数据绝对不能随机打乱，必须按时间顺序切分！)
    train_size = int(len(dummy_data) * 0.8)
    
    train_dataset = WeatherTimeSeriesDataset(dummy_data[:train_size], dummy_target[:train_size])
    val_dataset = WeatherTimeSeriesDataset(dummy_data[train_size:], dummy_target[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 注意: 仅在训练的批次级别打乱是安全的
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
