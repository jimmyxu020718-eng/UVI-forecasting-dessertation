###pearson correlation analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the new clear sky data
clear_sky_df = pd.read_csv('clear_sky_ghi_hkt.csv')

# Inspect clear_sky_df
print("Clear Sky Data Head:")
print(clear_sky_df.head())

# Assuming 'time' column exists based on the previous files pattern
if 'time' in clear_sky_df.columns:
    clear_sky_df['time'] = pd.to_datetime(clear_sky_df['time'])
    clear_sky_df.set_index('time', inplace=True)
else:
    # Check if there's another name for the time column
    print("Columns in clear_sky_ghi_hkt.csv:", clear_sky_df.columns.tolist())

# Resample clear sky to 1H (it might already be hourly, but this ensures alignment)
clear_sky_hourly = clear_sky_df.resample('1H').mean()

# 2. Load the previous merged data
merged_v3 = pd.read_csv('weather data and radiation_3.csv')
merged_v3['time'] = pd.to_datetime(merged_v3['time'])
merged_v3.set_index('time', inplace=True)

# 3. Final Merge
# Inner join to ensure we have all features for training
final_all_df = merged_v3.merge(clear_sky_hourly, left_index=True, right_index=True, how='inner')

# 4. Analysis & Correlation
correlation = final_all_df.corr()
print("\nCorrelation with Clear Sky GHI:")
target_col = [col for col in clear_sky_df.columns if 'ghi' in col.lower()][0]
print(correlation[target_col].sort_values(ascending=False))

# 5. Visualization: GHI vs Clear Sky GHI
plt.figure(figsize=(12, 6))
plt.plot(final_all_df.index[:168], final_all_df['GHI'][:168], label='Measured GHI', color='orange')
plt.plot(final_all_df.index[:168], final_all_df[target_col][:168], label='Clear Sky GHI', color='blue', linestyle='--')
plt.title('Comparison: Measured GHI vs Clear Sky GHI (One Week Sample)')
plt.legend()
plt.grid(True)
plt.savefig('ghi_vs_clearsky.png')

# 6. Final Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Final Feature Correlation Heatmap (All Features Combined)')
plt.tight_layout()
plt.savefig('final_all_features_heatmap.png')


### Scatter plot of UVI observations and various solar radiation data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Process Weather Data
weather_df = pd.read_csv('weather data and radiation.csv')
weather_df['time'] = pd.to_datetime(weather_df['time'])
weather_df.set_index('time', inplace=True)
weather_df = weather_df.drop(columns=['Unnamed: 5', 'Unnamed: 6'], errors='ignore')

weather_hourly = weather_df.resample('1H').agg({
    'wind_d': 'mean', 'windspd': 'mean', 'windspd_max': 'max',
    'temp': 'mean', 'rainfall': 'sum', 'GHI': 'mean',
    'DNI': 'mean', 'DHI': 'mean', 'uva': 'mean'
})

# 2. Process Cloud Data
cloud_df = pd.read_csv('cloud.csv')
cloud_df['time'] = pd.to_datetime(cloud_df['time'])
cloud_df.set_index('time', inplace=True)
cloud_df['cloud_amount'] = pd.to_numeric(cloud_df['cloud_amount'].replace('N', np.nan))
cloud_df['cloud_amount'] = cloud_df['cloud_amount'].interpolate(method='linear')

# 3. Process UV Data
uv_df = pd.read_csv('uv_5min.txt', sep='\s+', on_bad_lines='skip')
uv_df['time'] = pd.to_datetime(uv_df['YYYYMMDDhhmm'], format='%Y%m%d%H%M', errors='coerce')
uv_df = uv_df.dropna(subset=['time'])
uv_df.set_index('time', inplace=True)
uv_df['uv_5min'] = pd.to_numeric(uv_df['uv_5min'], errors='coerce')
uv_hourly = uv_df[['uv_5min']].resample('1H').mean()

# 4. Merge all
final_df = weather_hourly.merge(cloud_df, left_index=True, right_index=True, how='inner')
final_df = final_df.merge(uv_hourly, left_index=True, right_index=True, how='inner')

# 5. Create 2x2 Scatter plot grid (Without suptitle)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Subplot 1: uv_5min vs uva
sns.scatterplot(data=final_df, x='uva', y='uv_5min', ax=axes[0, 0], alpha=0.4, color='darkviolet')
axes[0, 0].set_title('uv_5min vs uva')
axes[0, 0].grid(True)

# Subplot 2: uv_5min vs GHI
sns.scatterplot(data=final_df, x='GHI', y='uv_5min', ax=axes[0, 1], alpha=0.4, color='orange')
axes[0, 1].set_title('uv_5min vs GHI')
axes[0, 1].grid(True)

# Subplot 3: uv_5min vs DNI
sns.scatterplot(data=final_df, x='DNI', y='uv_5min', ax=axes[1, 0], alpha=0.4, color='red')
axes[1, 0].set_title('uv_5min vs DNI')
axes[1, 0].grid(True)

# Subplot 4: uv_5min vs DHI
sns.scatterplot(data=final_df, x='DHI', y='uv_5min', ax=axes[1, 1], alpha=0.4, color='blue')
axes[1, 1].set_title('uv_5min vs DHI')
axes[1, 1].grid(True)

# Removed plt.suptitle to keep the top clean as requested
plt.tight_layout()
plt.savefig('uv_scatter_grid.png', dpi=300)
print("Image updated successfully without the main title.")