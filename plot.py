import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pandas as pd
import numpy as np
from func import ema, signals
from bot import trade_plot
import mplfinance as mpf
from decimal import Decimal
from bokeh.plotting import figure, show

def ema_plot(api, timestep, window_l, window_h, original_curves=False):
    h = api.history_highs(timestep)
    l = api.history_lows(timestep)
    #c = mocktrade(timestep=timestep, w_l=window_l, w_h=window_h, timeframe=api.history_length(), api=api, verbose=True)
    e_l = ema(l, window_l)
    e_h = ema(h, window_h)
    s = signals(e_l, e_h) * timestep
    print(l.shape)
    print(h.shape)
    print(s[-1])
    mid_open = np.mean(np.array([l[s[:,0] // timestep], h[s[:,0] // timestep]]), axis=0)
    mid_close = np.mean(np.array([l[s[:,1] // timestep], h[s[:,1] // timestep]]), axis=0)

    opentrade_closeprice = api.close_history_float()[s[:,0]]
    closetrade_closeprice = api.close_history_float()[s[:,1]]

    plt.plot(np.arange(0, h.shape[0]) * timestep, ema(h, window_h), color=(0, 0.5, 0.2))
    plt.plot(np.arange(0, l.shape[0]) * timestep, ema(l, window_l), color=(0.75, 0, 0))
    if original_curves:
        plt.plot(np.arange(0, api.history_length()), api.history_highs(1))
        plt.plot(np.arange(0, api.history_length()), api.history_lows(1))

    ##### close price that was traded on
    print(len(s))
    plt.scatter(s[:,0], opentrade_closeprice, c=[0.0, 1.0, 0.0], alpha=1)
    plt.scatter(s[:,1], closetrade_closeprice, c=[1.0, 0.0, 0.0], alpha=1)
    plt.show()

    profit_curve = np.array(trade_plot(api, s, 0.0007)[0]) * Decimal(1000.0)
    plt.plot(np.arange(0, profit_curve.shape[0]), profit_curve)
    plt.show()


def plot_curves(curves):
    max_length = max([c.shape[0] for c in curves])
    for curve in curves:
        plt.plot(np.arange(max_length - curve.shape[0], curve.shape[0]), curve)
    plt.show()

def create_dataframe(api, timestep, window_l, window_h):
    df = api.data_frame(timestep)

    h = api.history_highs(timestep)
    l = api.history_lows(timestep)
    e_l = ema(l, window_l)
    e_h = ema(h, window_h)
    s = signals(e_l, e_h) * timestep

    profit, open_long, close_long, open_short, close_short = trade_plot(api, s, 0.0007)

    df['open_long'] = [np.nan]*len(df)
    df['close_long'] = [np.nan]*len(df)
    df['open_short'] = [np.nan]*len(df)
    df['close_short'] = [np.nan]*len(df)

    go_long = s[:,0]
    end_long = s[:, 1]
    go_short = s[:,1] + 1
    end_short = s[:, 0] - 1

    # boundary checks
    end_long[-1] = min(api.history_length() - 2, end_long[-1])
    end_short[-1] = min(api.history_length() - 2, end_short[-1])
    go_long[-1] = min(api.history_length() - 2, go_long[-1])
    go_short[-1] = min(api.history_length() - 2, go_short[-1])

    df['open_long'][go_long // timestep] = open_long
    df['close_long'][end_long // timestep] = close_long
    df['open_short'][go_short // timestep] = open_short
    df['close_short'][end_short // timestep] = close_short

    #print(list(df['open_long'])[:5000])

    return df

# not animated yet lol
def animate_mocktrade(api, timestep, window_l, window_h, original_curves=True):
    df = create_dataframe(api, timestep, window_l, window_h)
    apds = [
        mpf.make_addplot(df['open_long'], type='scatter', markersize=100, marker='^', color="g"),
        mpf.make_addplot(df['close_long'], type='scatter', markersize=100, marker='^', color="r"),
        mpf.make_addplot(df['open_short'], type='scatter', markersize=100, marker='v', color="g"),
        mpf.make_addplot(df['close_short'], type='scatter', markersize=100, marker='v', color="r"),
    ]
    mpf.plot(df, type="candle", addplot=apds)

    
