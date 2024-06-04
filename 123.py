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
		<h1 style="color:white;text-align:center;">0050金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data(url):
    df = pd.read_pickle(url)
    return df

## 读取Pickle文件
start_date = st.text_input('選擇開始日期 (日期格式: 2019-01-01)', '2024-04-30')
end_date = st.text_input('選擇結束日期 (日期格式: 2019-01-01)', '2024-04-30')
start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
# 使用条件筛选选择时间区间的数据
df_original['time'] = pd.to_datetime(df_original['time'])
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]


###### (2) 轉化為字典 ######:
KBar_dic = df.to_dict()

KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open']=np.array(KBar_open_list)


KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
KBar_dic['time']=np.array(KBar_time_list)



KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low']=np.array(KBar_low_list)

KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high']=np.array(KBar_high_list)

KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close']=np.array(KBar_close_list)

KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume']=np.array(KBar_volume_list)

KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount']=np.array(KBar_amount_list)


######  (3) 改變 KBar 時間長度 (以下)  ########

Date = start_date.strftime("%Y-%m-%d")

st.subheader("設定一根 K 棒的時間長度(日)")
cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:日)', key="KBar_duration")
cycle_duration = int(cycle_duration)

KBar = indicator_forKBar_short.KBar(Date,cycle_duration * 1440)    ## 設定cycle_duration可以改成你想要的 KBar 週期


for i in range(KBar_dic['time'].size):
    
    #time = datetime.datetime.strptime(KBar_dic['time'][i],'%Y%m%d%H%M%S%f')
    time = KBar_dic['time'][i]
    #prod = KBar_dic['product'][i]
    open_price= KBar_dic['open'][i]
    close_price= KBar_dic['close'][i]
    low_price= KBar_dic['low'][i]
    high_price= KBar_dic['high'][i]
    qty =  KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    tag = KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

KBar_dic = {}


 ## 形成 KBar 字典 (新週期的):
KBar_dic['time'] =  KBar.TAKBar['time']   
#KBar_dic['product'] =  KBar.TAKBar['product']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] =  KBar.TAKBar['high']
KBar_dic['low'] =  KBar.TAKBar['low']
KBar_dic['close'] =  KBar.TAKBar['close']
KBar_dic['volume'] =  KBar.TAKBar['volume']
#
######  改變 KBar 時間長度 (以上)  ########



###### (4) 計算各種技術指標 ######
##### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic)


#####  (i) 移動平均線策略   #####
####  設定長短移動平均線的 K棒 長度:
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
#LongMAPeriod=st.number_input('輸入一個整數', key="Long_MA")
#LongMAPeriod=int(LongMAPeriod)
LongMAPeriod=st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
#ShortMAPeriod=st.number_input('輸入一個整數', key="Short_MA")
#ShortMAPeriod=int(ShortMAPeriod)
ShortMAPeriod=st.slider('選擇一個整數', 0, 100, 2)

#### 計算長短移動平均線
KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

#### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

#####  (ii) RSI 策略   #####
#### 順勢策略
### 設定長短 RSI 的 K棒 長度:
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod=st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod=st.slider('選擇一個整數', 0, 1000, 2)

### 計算 RSI指標長短線, 以及定義中線
## 假设 df 是一个包含价格数据的Pandas DataFrame，其中 'close' 是KBar週期收盤價
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle']=np.array([50]*len(KBar_dic['time']))

### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]


###### (5) 將 Dataframe 欄位名稱轉換  ###### 
KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]


###### (6) 畫圖 ######
st.subheader("畫圖")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from plotly.offline import plot
import plotly.offline as pyoff


##### K線圖, 移動平均線 MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)


##### K線圖, RSI
with st.expander("K線圖, 長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    #### include candlestick with rangeselector
    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    
    fig2.layout.yaxis2.showgrid=True
    st.plotly_chart(fig2, use_container_width=True)

# 計算KDJ指標
def calculate_kdj(df, n=9, m1=3, m2=3):
    df['low_min'] = df['Low'].rolling(window=n, min_periods=1).min()
    df['high_max'] = df['High'].rolling(window=n, min_periods=1).max()
    df['rsv'] = (df['Close'] - df['low_min']) / (df['high_max'] - df['low_min']) * 100
    df['k'] = df['rsv'].ewm(com=m1 - 1, adjust=False).mean()
    df['d'] = df['k'].ewm(com=m2 - 1, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    df.drop(['low_min', 'high_max', 'rsv'], axis=1, inplace=True)

# 將KDJ指標添加到KBar DataFrame中
calculate_kdj(KBar_df)

# 設置實單交易策略
def trading_strategy(df):
    df['buy_signal'] = (df['k'] > df['d']) & (df['k'].shift(1) < df['d'].shift(1))
    df['sell_signal'] = (df['k'] < df['d']) & (df['k'].shift(1) > df['d'].shift(1))
    # 根據實單交易策略執行相應的交易操作，這裡可以添加您的交易程式碼

# 執行交易策略
trading_strategy(KBar_df)

# 繪製包含KDJ指標和交易信號的圖表
with st.expander("K線圖, KDJ指標, 實單交易策略"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # K線圖
    fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    
    # KDJ指標
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['k'], mode='lines',line=dict(color='blue', width=2), name='K'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['d'], mode='lines',line=dict(color='red', width=2), name='D'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['j'], mode='lines',line=dict(color='green', width=2), name='J'), 
                  secondary_y=False)
    
    # 交易信號
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Close'], mode='markers', marker=dict(color='green', size=8), 
                               name='買入信號', customdata=KBar_df['buy_signal']), secondary_y=True)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Close'], mode='markers', marker=dict(color='red', size=8), 
                               name='賣出信號', customdata=KBar_df['sell_signal']), secondary_y=True)
    
    fig3.layout.yaxis2.showgrid=True
    st.plotly_chart(fig3, use_container_width=True)



# 計算布林通道指標
def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['std_dev'] = df['Close'].rolling(window=window).std()
    df['upper_band'] = df['MA'] + (num_std_dev * df['std_dev'])
    df['lower_band'] = df['MA'] - (num_std_dev * df['std_dev'])

# 添加布林通道指標到KBar DataFrame中
calculate_bollinger_bands(KBar_df)

# 繪製包含布林通道指標的圖表
with st.expander("K線圖, 布林通道"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # K線圖
    fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    
    # 布林通道
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['upper_band'], mode='lines',line=dict(color='orange', width=2), name='Upper Band'), 
                  secondary_y=False)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MA'], mode='lines',line=dict(color='blue', width=2), name='MA'), 
                  secondary_y=False)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['lower_band'], mode='lines',line=dict(color='purple', width=2), name='Lower Band'), 
                  secondary_y=False)
    
    fig4.layout.yaxis2.showgrid=True
    st.plotly_chart(fig4, use_container_width=True)










