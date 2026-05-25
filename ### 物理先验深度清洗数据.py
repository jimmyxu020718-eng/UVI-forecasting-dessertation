### 物理先验深度清洗数据

import numpy as np
import pandas as pd

def calculate_theoretical_clear_sky_ghi(df, lat=39.9, lon=116.4):
    """
    基于简化的 Haurwitz 晴空模型计算理论最大总水平面辐射 (Clear Sky GHI)。
    为深度网络提供绝对的物理边界先验。
    """
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear

    # 1. 太阳赤纬角 (Declination angle)计算
    declination = 23.45 * np.sin(
        2 * np.pi * (284 + df["day_of_year"]) / 365.25 * np.pi / 180
    )

    # 角度转弧度
    lat_rad = np.radians(lat)
    dec_rad = np.radians(declination)
    hr_rad = np.radians((df["hour"] - 12) * 15) # 时角

    # 2. 太阳高度角正弦值 (Sine of solar elevation angle)
    sin_elevation = np.sin(lat_rad) * np.sin(dec_rad) + \
                    np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hr_rad)
    sin_elevation = np.maximum(0, sin_elevation)  # 地平线以下物理截断

    # 3. Haurwitz 晴空模型辐射求解 (W/m²)
    # 物理公式: GHI_clear = 1098 * sin(h) * exp(-0.057 / sin(h))
    clear_sky_ghi = np.zeros(len(df))
    mask = sin_elevation > 0.01
    clear_sky_ghi[mask] = (
        1098 * sin_elevation[mask] * np.exp(-0.057 / sin_elevation[mask])
    )

    return clear_sky_ghi

    def apply_physical_prior_cleaning(df):
    """
    执行硬件级噪声过滤、夜间零掩码与物理特征解耦
    假设 df 已包含 'time', 'UVI', 'GHI' 等原始气象列
    """
    # ==========================================
    # Step A: 硬件饱和与量程溢出清洗 (Saturation Filtering)
    # ==========================================
    # 设定气象常识物理极值上限
    UVI_PHYSICAL_MAX = 20.0  
    GHI_PHYSICAL_MAX = 1500.0 

    # 将极值死点 (如传感器过载输出的 9999) 判定为 NaN，并用就近真实气象态插值修复
    df["UVI"] = np.where(df["UVI"] > UVI_PHYSICAL_MAX, np.nan, df["UVI"])
    df["GHI"] = np.where(df["GHI"] > GHI_PHYSICAL_MAX, np.nan, df["GHI"])
    df["UVI"] = df["UVI"].ffill().bfill()
    df["GHI"] = df["GHI"].ffill().bfill()

    # ==========================================
    # Step B: 夜间热噪声/暗电流零掩码 (Nighttime Zero-Masking)
    # ==========================================
    # 彻底抹除夜间光电传感器因温度变化产生的 0.02, 0.05 等非零伪影
    df["time_hour"] = df["time"].dt.hour
    is_night = (df["time_hour"] >= 19) | (df["time_hour"] <= 5)
    
    df.loc[is_night, "UVI"] = 0.0
    df.loc[is_night, "GHI"] = 0.0

    # ==========================================
    # Step C: 提取核心物理先验特征 —— 晴空指数 (Clear Sky Index)
    # ==========================================
    # 1. 调取天文理论辐射值
    df["GHI_theoretical_max"] = calculate_theoretical_clear_sky_ghi(df)

    # 2. 解耦云层消光系数 (kt = 实际监测辐射 / 绝对晴空辐射)
    # 引入 1e-3 避免微积分母除零错位
    df["Clear_Sky_Index_kt"] = df["GHI"] / (df["GHI_theoretical_max"] + 1e-3)

    # 3. 合法性越界截断 (处理云侧强反射导致的短暂超射)
    df["Clear_Sky_Index_kt"] = np.clip(df["Clear_Sky_Index_kt"], 0.0, 1.2)
    
    # 4. 再次确保夜间无异常特征输入网络
    df.loc[is_night, "Clear_Sky_Index_kt"] = 0.0

    return df