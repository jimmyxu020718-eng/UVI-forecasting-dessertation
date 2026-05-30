### SMLP

import torch
import torch.nn as nn

class DualAttnBlock(nn.Module):
    """时间注意力 + 跨特征注意力块"""
    def __init__(self, seq_len, feature_dim, num_heads=2, dropout=0.1):
        super().__init__()
        # ---------- 时间自注意力 ----------
        self.norm_time = nn.LayerNorm(feature_dim)          # 对每个时间步的特征向量归一化
        self.time_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True               # 输入形状 (B, seq_len, feature_dim)
        )

        # ---------- 跨特征自注意力 ----------
        self.norm_feat = nn.LayerNorm(seq_len)              # 转置后归一化最后一维（时间序列长度）
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=seq_len,             # 每个“特征”用所有时间步的值作为向量
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True               # 输入形状 (B, feature_dim, seq_len)
        )

    def forward(self, x):
        # x: (B, T, F)  T=seq_len, F=feature_dim

        # 1. 时间自注意力
        residual = x
        x_norm = self.norm_time(x)
        time_out, _ = self.time_attn(x_norm, x_norm, x_norm)
        x = residual + time_out

        # 2. 跨特征自注意力
        residual = x
        x_feat = x.transpose(1, 2)                  # (B, F, T)
        x_feat = self.norm_feat(x_feat)
        feat_out, _ = self.feat_attn(x_feat, x_feat, x_feat)
        feat_out = feat_out.transpose(1, 2)         # 转回 (B, T, F)
        x = residual + feat_out

        return x


class SMLPDualAttnModel(nn.Module):
    """双重注意力 SMLP 预测模型"""
    def __init__(self, seq_len=96, pred_len=24, feature_dim=12,
                 num_blocks=3, attn_heads=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 堆叠多个双重注意力块
        self.blocks = nn.ModuleList([
            DualAttnBlock(seq_len, feature_dim, attn_heads, dropout)
            for _ in range(num_blocks)
        ])

        # 预测头：平均池化 + 最后一步 拼接后映射到预测长度
        self.proj = nn.Linear(feature_dim * 2, 256)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len)
        )

    def forward(self, x):
        # x: (B, T, F)  假设第0列为历史真实值，第1列为晴空上限
        hidden = x
        for block in self.blocks:
            hidden = block(hidden)

        # 提取全局趋势和最新状态
        avg_pool = hidden.mean(dim=1)           # (B, F)
        last_step = hidden[:, -1, :]            # (B, F)
        combined = torch.cat([avg_pool, last_step], dim=-1)  # (B, 2F)

        out = self.proj(combined)
        pred = self.head(out)                   # (B, pred_len)

        # 物理约束：非负且不超过晴空上限
        clear_sky = x[:, -1, 1]                 # (B,)
        pred = torch.clamp(pred, min=0, max=clear_sky.unsqueeze(1))
        return pred


# ----- 测试样例 -----
if __name__ == "__main__":
    dummy = torch.randn(32, 96, 12)
    model = SMLPDualAttnModel(seq_len=96, pred_len=24, feature_dim=12,
                              num_blocks=3, attn_heads=2)
    output = model(dummy)
    print(f"输入维度: {dummy.shape}")
    print(f"输出维度: {output.shape}")   # 应为 (32, 24)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {params/1e6:.4f} M")
