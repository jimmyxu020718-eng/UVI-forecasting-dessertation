### trainer

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

# ========================================================
# 1. 时序滑动窗口数据集
# ========================================================
class WeatherTimeSeriesDataset(Dataset):
    """
    每个样本：历史 seq_len 小时所有特征 → 未来 pred_len 小时目标值
    """
    def __init__(self, data_matrix, target_matrix, seq_len=96, pred_len=24):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.valid_length = len(data_matrix) - seq_len - pred_len + 1
        assert self.valid_length > 0, "数据长度不足以构建一个窗口"
        self.x = torch.FloatTensor(data_matrix)
        self.y = torch.FloatTensor(target_matrix)

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        seq_x = self.x[idx : idx + self.seq_len, :]          # (seq_len, feature_dim)
        seq_y = self.y[idx + self.seq_len : idx + self.seq_len + self.pred_len, 0]  # (pred_len,)
        return seq_x, seq_y


# ========================================================
# 2. 训练函数（含 warmup + 余弦退火 + 早停）
# ========================================================
def train_model(model, train_loader, val_loader, epochs=50, device='cuda',
                lr=1e-3, weight_decay=1e-4, patience=7, warmup_epochs=5):
    model = model.to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)      # Huber Loss，beta 可改为 2.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- 学习率调度：warmup + 余弦退火 ----
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性增长
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火从 warmup 结束到最终
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    counter = 0
    best_model_weights = None

    print(f"🚀 开始训练 (warmup={warmup_epochs} epochs, patience={patience})")

    for epoch in range(epochs):
        # ---------- 训练 ----------
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)                     # (batch, pred_len)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪（注意力模型建议 2.0）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️  早停触发（{patience} epoch 无改善），停止训练")
                break

    # 恢复最佳权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    print(f"✅ 训练完成，最佳验证损失: {best_val_loss:.4f}")
    return model


# ========================================================
# 3. 辅助函数：测试集评估
# ========================================================
@torch.no_grad()
def evaluate(model, data_loader, device='cuda'):
    model.eval()
    criterion = nn.SmoothL1Loss(beta=1.0)
    total_loss = 0.0
    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        total_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
    return total_loss / len(data_loader.dataset)


# ========================================================
# 4. 使用示例（与你的数据划分对接）
# ========================================================
if __name__ == "__main__":

    data = np
    dummy_target = np

    train_dataset = WeatherTimeSeriesDataset(dummy_data[:3000], dummy_target[:3000])
    val_dataset   = WeatherTimeSeriesDataset(dummy_data[3000:4000], dummy_target[3000:4000])
    test_dataset  = WeatherTimeSeriesDataset(dummy_data[4000:], dummy_target[4000:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型实例化（需提前定义 SMLPDualAttnModel）
    # model = SMLPDualAttnModel(...)
    # model = train_model(model, train_loader, val_loader, epochs=50, device='cuda')

    # 测试
    # test_loss = evaluate(model, test_loader, device='cuda')
    # print(f"📊 测试集损失: {test_loss:.4f}")
