import os
import pandas as pd
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from rlenv.StockTradingEnv0 import StockTradingEnv
from RLDataset import get_data_pack

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False


def stock_trade(df_train, df_test):
    day_profits = ([], [])

    # The algorithms require a vectorized environment to run
    env = make_vec_env(lambda: StockTradingEnv(df_train))

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e4))

    env = make_vec_env(lambda: StockTradingEnv(df_test))

    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, ternimated, info = env.step(action)
        env.render()
        profit = info[0]['daily_pnl']
        day_profits[0].append(df_test['date'][i])
        day_profits[1].append(profit)
        if ternimated:
            break
    return day_profits

def test_a_stock_trade(stock_code):

    fetch_data_beg = 20140101
    fetch_data_end = 20200101

    df_d01 = get_data_pack(tp_beg=f'{fetch_data_beg} 00:00:00.000', tp_end=f'{fetch_data_end} 17:00.00.000')
    df_d01['avgprice'].fillna(df_d01['close'], inplace=True)
    df_d01 = df_d01[df_d01['ticker'] == stock_code]
    bdt_train, edt_train, bdt_test, edt_test = 20140101, 20151231, 20160101, 20200101
    df_train = df_d01.query(f'date>="{bdt_train}" & date<="{edt_train}"').reset_index(drop=True)
    df_test = df_d01.query(f'date>="{bdt_test}" & date<="{edt_test}"').reset_index(drop=True)

    daily_profits = stock_trade(df_train=df_train, df_test=df_test)
    fig, ax = plt.subplots()
    ax.plot(daily_profits[0], daily_profits[1], marker='o', label=stock_code,  mfc='orange')
    ax.grid()
    plt.xlabel('date')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')

if __name__ == '__main__':
    test_a_stock_trade('600036')
