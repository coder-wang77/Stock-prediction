import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

def download_stock_data(symbol, start='2016-01-01', end='2025-01-01'):
    stock = yf.download(symbol, start=start, end=end)
    return stock
def stock_add_indicator(df, sma, lma):
    df['SMA'] = df.Close.rolling(window=sma).mean()
    df['LMA'] = df.Close.rolling(window=lma).mean()
    return df
def stock_buy_sell(df, target_price_per, stop_loss_per, holding_period):
    df['Buy'] = 0
    for i in range(len(df) - holding_period):
        current_price = df[('Open', 'AMD')].iloc[i]
        for j in range(holding_period):
            if i+j>=len(df):
                break
            max_price_gain = df[('High', 'AMD')].iloc[i+j] - current_price
            max_price_lose = current_price - df[('Low', 'AMD')].iloc[i+j]
            if max_price_lose >= current_price * stop_loss_per:
                df.at[df.index[i], 'Buy'] = 0
                break
            if max_price_gain >= current_price * target_price_per:
                df.at[df.index[i], 'Buy'] = 1
                break
    return df

# Get a cleaned dataframe
df = download_stock_data('AMD').reset_index()
df = stock_add_indicator(df,5,200)
df_cleaned = df.dropna()
df_cleaned = stock_buy_sell(df_cleaned,0.01,0.003,5)
y = df_cleaned['Buy']
# Get features for volume, sma, lma
X = df_cleaned.iloc[:,5:-1]
# Scale the features
X_scaled = StandardScaler().fit_transform(X)
fig, ax = plt.subplots(1,3)
ax[0].hist(X['Volume'])
ax[0].set_title('Volume')
ax[1].hist(X['SMA'])
ax[1].set_title('SMA')
ax[2].hist(X['LMA'])
ax[2].set_title('LMA')
plt.show()
# Get test and train set
x_train, x_test, y_train, y_test = train_test_split(X_scaled,y,train_size=0.8, test_size=0.2, random_state=10)
# Using a LogisticRegression model to get Classifications (buy signal)
lr = LogisticRegression().fit(x_train,y_train)
y_pred = lr.predict(x_test)
corr_table = pd.crosstab(y_test, y_pred)
print(corr_table)
print(f'accuracy score is: {accuracy_score(y_test, y_pred)}')
print(f'recall score is: {recall_score(y_test, y_pred)}')
print(f'precision score is: {precision_score(y_test, y_pred)}')