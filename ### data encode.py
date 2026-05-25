### data encode

import numpy as np
import pandas as pd

def encode_cyclical_features(df, time_col_is_index=True):
    """
    对时序数据进行周期性正余弦编码。
    为模型提供连续的时间上下文特征。
    """
    print("开始执行时序特征周期性编码...")
    
    # 确保操作不改变原始数据
    df_encoded = df.copy()
    
    # 提取时间对象 (如果 time 是 index，则提取出来)
    if time_col_is_index:
        time_series = df_encoded.index
    else:
        time_series = pd.to_datetime(df_encoded['time'])
    
    # 1. 提取时间维度的基准数值
    hours = time_series.hour
    day_of_year = time_series.dayofyear
    months = time_series.month
    
    # 获取当年的总天数 (处理闰年 365/366)
    days_in_year = time_series.is_leap_year.map({True: 366, False: 365})
    
    # ==========================================
    # 2. 核心：日周期编码 (Diurnal Cycle)
    # ==========================================
    # 捕捉 24 小时内的波动节律 (对 UVI 和 GHI 最关键)
    df_encoded['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df_encoded['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    
    # ==========================================
    # 3. 核心：年周期编码 (Annual/Seasonal Cycle)
    # ==========================================
    # 捕捉一年中由于太阳直射点移动带来的季节性衰减规律
    df_encoded['day_year_sin'] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df_encoded['day_year_cos'] = np.cos(2 * np.pi * day_of_year / days_in_year)
    
    # ==========================================
    # 4. 可选：月度宏观特征 (Monthly Level)
    # ==========================================
    df_encoded['month_sin'] = np.sin(2 * np.pi * months / 12.0)
    df_encoded['month_cos'] = np.cos(2 * np.pi * months / 12.0)
    
    # 打印编码后的特征检查
    print("编码完成！新增的连续时间特征包括:")
    print("['hour_sin', 'hour_cos', 'day_year_sin', 'day_year_cos', 'month_sin', 'month_cos']")
    
    return df_encoded

# ==========================================
# 调试与本地测试
# ==========================================
if __name__ == "__main__":
    # 假设你已经运行了上一步的代码，得到了清洗后的 df_hourly
    # 这里我们快速生成一个包含 48 小时的 dummy DataFrame 模拟一下
    time_idx = pd.date_range(start='2026-12-30 00:00:00', periods=48, freq='1H')
    df_sim = pd.DataFrame({
        'UVI': np.random.uniform(0, 10, 48),
        'Clear_Sky_Index_kt': np.random.uniform(0, 1, 48)
    }, index=time_idx)
    
    # 调用编码函数
    df_ready_for_model = encode_cyclical_features(df_sim, time_col_is_index=True)
    
    # 查看跨越零点（23:00 -> 00:00）和跨年（12.31 -> 01.01）时的平滑过渡
    print("\n观察跨日零点时 (23:00 到 00:00) 编码特征的连续性:")
    print(df_ready_for_model[['hour_sin', 'hour_cos', 'day_year_sin']].iloc[22:26])