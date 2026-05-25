### plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 0. 全局学术排版样式配置 (IEEE Style)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#b0b0b0'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#f0f0f0'

# 定义符合国际期刊审稿人审美的标准色系（高级莫兰迪/安全色）
COLOR_PALETTE = {
    'Proposed': '#1f78b4',      # 经典深蓝 (代表本研究提出的 Stacked MLP)
    'TFT': '#d95f02',           # 铁锈橙
    'TSMixer': '#4daf4a',       # 橄榄绿
    'Autoformer': '#984ea3',    # 优雅紫
    'DLinear': '#ff7f00'         # 亮橙
}

# ==============================================================================
# 函数 1: 24小时周期绝对残差演变图 (Diurnal Residuals)
# ==============================================================================
def plot_diurnal_residuals(test_hours, y_true, y_preds_dict, save_path='Figure_1_Diurnal_Residuals.png'):
    """
    分析误差在一天24小时时间维度上的动态演变过程（对应 6.3 节）。
    test_hours: 测试集对应的24小时制小时数组 (形状与 y_true 相同)
    y_true: 真实 UVI 数组
    y_preds_dict: 字典，格式为 { '模型名称': 预测值数组 }
    """
    print("正在绘制：24小时周期残差演变图...")
    hours = np.arange(24)
    
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    
    # 遍历字典中的每一个对比模型，计算每小时的平均绝对误差(MAE)
    for model_name, y_pred in y_preds_dict.items():
        absolute_errors = np.abs(y_true - y_pred)
        
        # 按小时(0-23)打包聚合计算平均值
        hourly_mae = [np.mean(absolute_errors[test_hours == h]) for h in hours]
        
        # 匹配颜色与线型
        color = COLOR_PALETTE.get(model_name, '#757575')
        linewidth = 2.5 if 'Proposed' in model_name or 'MLP' in model_name else 1.5
        linestyle = '-' if 'Proposed' in model_name or 'MLP' in model_name else '--'
        marker = 's' if 'Proposed' in model_name or 'MLP' in model_name else 'o'
        
        ax.plot(hours, hourly_mae, label=model_name, color=color, 
                linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=4)

    ax.set_xlabel('Hour of Day', fontweight='bold', labelpad=8)
    ax.set_ylabel('Mean Absolute Error (UVI)', fontweight='bold', labelpad=8)
    ax.set_title('Comprehensive 24-Hour Diurnal Absolute Residuals Comparison', pad=15, fontweight='bold', fontsize=12)
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlim([-0.5, 23.5])
    ax.legend(loc='upper left', frameon=True, shadow=False, facecolor='white', edgecolor='#e0e0e0', ncol=2)
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"成功保存至: {save_path}")

# ==============================================================================
# 函数 2: 全局误差分布与统计鲁棒性箱线图 (Global Error Boxplot)
# ==============================================================================
def plot_error_boxplot(y_true, y_preds_dict, save_path='Figure_2_Error_Boxplot.png'):
    """
    从统计学维度直观揭示各个模型对抗异常值的鲁棒性（对应 6.4 节）。
    """
    print("正在绘制：全局误差分布箱线图...")
    
    # 构造适合 seaborn 排版的数据框
    error_data = []
    model_names = []
    
    for model_name, y_pred in y_preds_dict.items():
        abs_error = np.abs(y_true - y_pred)
        error_data.append(abs_error)
        model_names.extend([model_name] * len(abs_error))
        
    df_box = pd.DataFrame({
        'Absolute Error (UVI)': np.concatenate(error_data),
        'Model Architecture': model_names
    })

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # 根据当前传入的模型动态匹配调色板
    current_palette = [COLOR_PALETTE.get(name, '#757575') for name in y_preds_dict.keys()]
    
    sns.boxplot(x='Model Architecture', y='Absolute Error (UVI)', data=df_box, ax=ax,
                palette=current_palette, width=0.4,
                flierprops=dict(marker='o', markerfacecolor='#9e9e9e', markersize=3, markeredgecolor='none', alpha=0.3))

    ax.set_ylabel('Absolute Prediction Error (UVI)', fontweight='bold', labelpad=8)
    ax.set_xlabel('Model Architecture', fontweight='bold', labelpad=8)
    ax.set_title('Global Prediction Error Distribution and Outlier Analysis', pad=15, fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"成功保存至: {save_path}")

# ==============================================================================
# 函数 3: 2x2 细分流派一对一回归散点解构矩阵 (Head-to-Head Scatter Matrix)
# ==============================================================================
def plot_head_to_head_scatter(y_true, mlp_pred, baseline_preds_dict, save_path='Figure_3_HeadToHead_Scatter_Matrix.png'):
    """
    将本研究模型与四大流派基准模型进行 2x2 正面对决散点解构（对应 6.5 节）。
    mlp_pred: 本研究提出的 Stacked MLP 预测数组
    baseline_preds_dict: 其他 4 个 Baseline 模型的预测字典 (元素必须正好为 4 个)
    """
    print("正在绘制：2x2 细分流派一对一回归散点矩阵...")
    baselines = list(baseline_preds_dict.keys())
    if len(baselines) != 4:
        raise ValueError("为了完美的 2x2 排版，baseline_preds_dict 必须包含且仅包含 4 个基准模型！")
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
    axs = axs.flatten()
    
    subplots_labels = ['(a) Proposed SMLP vs. ', '(b) Proposed SMLP vs. ', '(c) Proposed SMLP vs. ', '(d) Proposed SMLP vs. ']

    for i, base_name in enumerate(baselines):
        ax = axs[i]
        y_base_pred = baseline_preds_dict[base_name]
        color_base = COLOR_PALETTE.get(base_name, '#757575')
        
        # 1. 绘制基准模型的底层散点 (低对比度透明度，防止遮挡)
        ax.scatter(y_true, y_base_pred, alpha=0.25, color=color_base, edgecolors='none', s=25, label=f'{base_name}')
        
        # 2. 叠加本研究模型的预测散点 (高对比度深蓝)
        ax.scatter(y_true, mlp_pred, alpha=0.35, color=COLOR_PALETTE['Proposed'], edgecolors='none', s=25, label='Stacked MLP (Ours)')
        
        # 3. 绘制理想对角线 y = x
        max_val = max(np.max(y_true), 13.0)
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.2, alpha=0.7, label='Perfect Prediction (y = x)')
        
        # 界面美化设置
        ax.set_xlabel('Ground Truth UVI', fontweight='bold', fontsize=10)
        ax.set_ylabel('Predicted UVI', fontweight='bold', fontsize=10)
        ax.set_title(f"{subplots_labels[i]}{base_name}", pad=10, fontweight='bold', fontsize=11)
        ax.set_xlim([0, max_val + 0.5])
        ax.set_ylim([0, max_val + 0.5])
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"成功保存至: {save_path}")