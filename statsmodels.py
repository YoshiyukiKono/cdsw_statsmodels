import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

row_data = pd.read_csv('tokyo2016.csv', index_col=0)

df = row_data.reset_index()
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
df['power'] = df['power'].values.astype(int)
daily_data = df.groupby('date')['power'].sum().reset_index() #元データが1時間ごとなので日毎のデータにする
daily_data = daily_data.set_index(['date'])
daily_data.plot()

import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(daily_data.values, freq=7)
res.plot()

# 季節成分を除いたトレンド
trend = res.trend
trend = pd.DataFrame({'trend': trend, 'date':daily_data.index})
trend['date'] = pd.to_datetime(trend['date'], format='%Y-%m-%d')
trend = trend.set_index(['date'])
trend = trend.plot()

# 曜日成分

seasonal_data = res.seasonal
seasonal = pd.DataFrame({'seasonal': seasonal_data, 'date':daily_data.index})
seasonal['date'] = pd.to_datetime(seasonal['date'], format='%Y-%m-%d')
seasonal['weekday'] = seasonal['date'].dt.dayofweek
seasonal[0:7]

# 時間帯別

df = row_data.reset_index()
morning_power = df[df['time'] == '8:00'].reset_index()
morning_power['date'] = pd.to_datetime(morning_power['date'], format='%Y/%m/%d')
morning_power['power'] = morning_power['power'].values.astype(int)
morning_data = morning_power.groupby('date')['power'].sum().reset_index()
morning_data = morning_data.set_index(['date'])
res = sm.tsa.seasonal_decompose(morning_data.values, freq=7)
res.plot()

