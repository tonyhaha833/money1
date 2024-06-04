# 載入必要模組
import os
import numpy as np
import indicator_f_Lo2_short, datetime, indicator_forKBar_short
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 

###### (1) 開始設定 ######
html_temp = """
    <div style="background-color:#808080;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
    <h2 style="color:white;text-align:center;">Financial Dashboard </h2>
    </div>
    """
stc.html(html_temp)
## 读取Pickle文件
@st.cache_data(ttl=3600, show_spinner="正在加载数据")
def load_data(url):
    df = pd.read_pickle(url)
    return df
df_original = load_data('testdata.pkl')

df_original = df_original.drop('Unnamed: 0', axis=1)

##### 選擇資料區間
st.subheader("選擇開始與結束的日期, 區間:2022-01-03 至 2022-11-18")
start_date = st.text_input('選擇開始日期 (日期格式: 2022-01-03)', '2022-01-03')
end_date = st.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')
start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

###### (2) 轉化為字典 ######:
KBar_dic = df.to_dict()
KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open'] = np.array(KBar_open_list)
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] 
KBar_dic['time'] = np.array(KBar_time_list)
KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low'] = np.array(KBar_low_list)
KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high'] = np.array(KBar_high_list)
KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close'] = np.array(KBar_close_list)
KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume'] = np.array(KBar_volume_list)
KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount'] = np.array(KBar_amount_list)

######  (3) 改變 KBar 時間長度 (以下)  ########
Date = start_date.strftime("%Y-%m-%d")
st.subheader("設定一根 K 棒的時間長度(分鐘)")
time_units = ["分鐘", "小時", "天"]
selected_unit = st.selectbox("选择时间单位", time_units)

if selected_unit == "分鐘":
    unit_conversion = 1
elif selected_unit == "小時":
    unit_conversion = 60
else:
    unit_conversion = 1440

cycle_duration = st.number_input('输入一根 K 棒的时间长度（单位：{}）'.format(selected_unit), value=1, key="KBar_duration") * unit_conversion
cycle_duration = int(cycle_duration)
KBar = indicator_forKBar_short.KBar(Date, cycle_duration)

for i in range(KBar_dic['time'].size):
    time = KBar_dic['time'][i]
    open_price = KBar_dic['open'][i]
    close_price = KBar_dic['close'][i]
    low_price = KBar_dic['low'][i]
    high_price = KBar_dic['high'][i]
    qty = KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    tag = KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
KBar_dic = {}
KBar_dic['time'] = KBar.TAKBar['time']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] = KBar.TAKBar['high']
KBar_dic['low'] = KBar.TAKBar['low']
KBar_dic['close'] = KBar.TAKBar['close']
KBar_dic['volume'] = KBar.TAKBar['volume']

###### (4) 計算各種技術指標 ######
KBar_df = pd.DataFrame(KBar_dic)

#####  (i) 移動平均線策略   #####
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
LongMAPeriod = st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
ShortMAPeriod = st.slider('選擇一個整數', 0, 100, 2)

KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

#####  (ii) RSI 策略   #####
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2)

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

###### (5) 將 Dataframe 欄位名稱轉換  ###### 
KBar_df.columns = [i[0].upper() + i[1:] for i in KBar_df.columns]

###### (6) 畫圖 ######
st.subheader("畫圖")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyoff

##### K線圖, 移動平均線 MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['open'], high=KBar_df['high'],
                    low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig1.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig1.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines', name=f'{LongMAPeriod} K 長移動平均線', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines', name=f'{ShortMAPeriod} K 短移動平均線', line=dict(color='red')))
    fig1.update_layout(title='K線圖, 移動平均線')
    fig1.update_yaxes(title_text="價格", secondary_y=True)
    fig1.update_yaxes(title_text="交易量", secondary_y=False)
    st.plotly_chart(fig1)

##### RSI 圖
with st.expander("RSI"):
    fig2 = make_subplots()
    fig2.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', name=f'{LongRSIPeriod} K 長RSI', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', name=f'{ShortRSIPeriod} K 短RSI', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_Middle'][last_nan_index_RSI+1:], mode='lines', name='RSI 中間線', line=dict(color='green')))
    fig2.update_layout(title='RSI')
    st.plotly_chart(fig2)

##### K線圖, RSI
with st.expander("K線圖, RSI"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    fig3.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', name=f'{LongRSIPeriod} K 長RSI', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', name=f'{ShortRSIPeriod} K 短RSI', line=dict(color='red')))
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_Middle'][last_nan_index_RSI+1:], mode='lines', name='RSI 中間線', line=dict(color='green')))
    fig3.update_layout(title='K線圖, RSI')
    fig3.update_yaxes(title_text="價格", secondary_y=True)
    fig3.update_yaxes(title_text="交易量", secondary_y=False)
    st.plotly_chart(fig3)

##### 總K線圖, RSI, MA
with st.expander("K線圖, RSI, MA"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Candlestick(x=KBar_df['time'],
                    open=KBar_df['open'], high=KBar_df['high'],
                    low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig4.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines', name=f'{LongMAPeriod} K 長移動平均線', line=dict(color='blue')))
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines', name=f'{ShortMAPeriod} K 短移動平均線', line=dict(color='red')))
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', name=f'{LongRSIPeriod} K 長RSI', line=dict(color='blue')))
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', name=f'{ShortRSIPeriod} K 短RSI', line=dict(color='red')))
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_Middle'][last_nan_index_RSI+1:], mode='lines', name='RSI 中間線', line=dict(color='green')))
    fig4.update_layout(title='K線圖, RSI, 移動平均線')
    fig4.update_yaxes(title_text="價格", secondary_y=True)
    fig4.update_yaxes(title_text="交易量", secondary_y=False)
    st.plotly_chart(fig4)
