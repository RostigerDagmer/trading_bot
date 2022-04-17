from api import *
from decimal import Decimal
from func import ema, signals
from bot import trade_history, static_trade_history
from log import Database
import tables
import os

def instance_c_market_cap_response():
    data = {'habe': 'dere'}
    r = CMarketCapResponse(data)
    print(r.data == data)
    return

def connect_c_market_cap():
    api = CMarketCapAPI(url, API_KEY)
    api('ETH', '1')
    print(api.get_price().json())

# random forest best estimate: timestep=55, w_l=2, w_h=68
# random forest hf estimate: timestep=10, w_l=3, w_h=17
# debiased forest best estimate timestep=278, w_l=2657, w_h=88
# parallel forest best estimate timestep=1, w_l=2, w_h=9
# parallel forest best estimate timestep=10, w_l=2, w_h=21
# parallel forest best estimate timestep=14, w_l=2, w_h=61
# parallel forest best estimate timestep=14, w_l=2, w_h=43
# exaustive best estimate 1: timestep=7, w_l=2, w_h=34
# exaustive best estimate 2: timestep=4, w_l=2, w_h=39
# max lines: 613813
def mocktrade(timestep=4, trading_fee=0.0007, capital=Decimal('1000.0'), w_l=100, w_h=97, timeframe=12000, api=None, verbose=False):
    l = api.history_lows(timestep, timeframe)
    h = api.history_highs(timestep, timeframe)
    e_l = ema(l, w_l)
    e_h = ema(h, w_h)
    s = signals(e_l, e_h) * timestep
    if len(s) <= 0:
        # no signals found should be penalized
        #print('no valid signals')
        return Decimal('-10e15')
    C = trade_history(api, s, trading_fee)
    if verbose:
        print(f"invested: {capital}")
        print(f"current value: {capital * C}")
        print(f"profit: {capital * C - capital}")
    return capital * C - capital

def filter_h(f, a, timestep, timeframe_start, timeframe_end):
    return f(a[(timeframe_start // timestep) * timestep:(timeframe_end // timestep) * timestep].reshape([(timeframe_end // timestep) - (timeframe_start // timestep), timestep]))

''' note that this function trades accuracy for speed and accepts floating point error'''
def vec_mocktrade(timesteps: np.ndarray, 
                    w_l: np.ndarray, 
                    w_h: np.ndarray, 
                    timeframes_end=np.ndarray, 
                    timeframes_start=None, 
                    trading_fee=0.0007, 
                    capital=1000.0, 
                    lows=None, 
                    highs=None, 
                    history=None, 
                    threadpool=None, 
                    verbose=False):
    #assert history and lows and highs
    n_dim = timesteps.shape[0]
    #initialize trading fee vector
    trading_fees = np.ones(n_dim)
    trading_fees.fill(trading_fee)
    if not timeframes_start:
        timeframes_start = np.zeros(n_dim, dtype=int)

    def f(vParam):
        timestep = vParam[0]
        w_l = vParam[1]
        w_h = vParam[2]
        timeframe_start = vParam[3]
        timeframe_end = vParam[4]
        l = filter_h(lambda x: np.amin(x, axis=1), lows, timestep, timeframe_start, timeframe_end)
        h = filter_h(lambda x: np.amax(x, axis=1), highs, timestep, timeframe_start, timeframe_end)
        #print(f'timestep: {timestep}')
        #print(l)
        #print(h)
        e_l = ema(l, w_l)
        e_h = ema(h, w_h)
        s = signals(e_l, e_h) * timestep
        #print(s)
        if len(s) <= 0:
            # no signals found should be penalized
            return -1000000000000000.0
        C = static_trade_history(history[timeframe_start:timeframe_end], s, trading_fee)
        return capital * C - capital

    params = np.transpose(np.array([timesteps, w_l, w_h, timeframes_start, timeframes_end]))
    
    if threadpool:
        return threadpool.map(f, params)
    return list(map(f, params))


def mocktrade_history():
    api = MockAPI(datapath='data/Bitstamp_ETHUSD_2021_minute.csv', readlines=613813)
    print('timestep=1, w_l=2, w_h=9')
    mocktrade(timestep=1, w_l=2, w_h=9, timeframe=613813, api=api, verbose=True)
    print('timestep=10, w_l=3, w_h=17')
    mocktrade(timestep=10, w_l=3, w_h=17, timeframe=613813, api=api, verbose=True)
    print('timestep=3, w_l=6, w_h=16')
    mocktrade(timestep=3, w_l=6, w_h=16, timeframe=613813, api=api, verbose=True)

def test_vec_mocktrade():
    timesteps = np.array([10, 3, 14, 13], dtype=int)
    wls = np.array([3, 6, 2, 2], dtype=int)
    whs = np.array([17, 16, 43, 24], dtype=int)
    timeframes = np.array([613813, 613813, 613813, 613813])

    api = MockAPI(datapath='data/Bitstamp_ETHUSD_2021_minute.csv', readlines=613813)
    lows = api.history_lows(1)
    highs = api.history_highs(1)
    close_history = api.close_history_float()

    res = vec_mocktrade(timesteps, wls, whs, timeframes_end=timeframes, timeframes_start=None, lows=lows, highs=highs, history=close_history)
    print(res)

def test_signals():
    to_ = 20000
    from_ = 1000
    step = 1000
    timeframes = list(range(from_, to_, step))
    api = MockAPI(datapath='data/Bitstamp_ETHUSD_2021_minute.csv', timeframe=to_)
    lows = api.history_lows()
    highs = api.history_highs()

def test_table_creation():
    db = Database()
    t = np.dtype([("trades", "S16"), 
                    ("ema_low", np.float64), 
                    ("ema_high", np.float64), 
                    ("capital", np.float64)
                    ])
    db.create_table("bot", "readout", t)
    db.show()

def test_data_write():
    db = Database()
    t = np.dtype([("trades", "S16"), 
                    ("ema_low", np.float64), 
                    ("ema_high", np.float64), 
                    ("capital", np.float64)
                    ])
    d = {'ema_low': np.array([1,2,3], dtype=float),
         'ema_high':  np.array([2,3, 4], dtype=float),
         "capital":  np.array([6,7,8], dtype=float),
         "trades": np.array(['long', 'short', 'long'])}
         
    db.create_table("bot", "readout", t)
    db.write_data("bot", "readout", d)
    db.sync()
    db.show()

def test_data_read():
    d = os.path.join(os.getcwd(), 'tmp')
    f = tables.open_file(os.path.join(d, os.listdir(d)[-1]), driver="H5FD_CORE")
    print(f)

tests = {
    #'instance_c_market_cap_response': instance_c_market_cap_response,
    #'mocktrade_history': mocktrade_history,
    'vec_mocktrade': test_vec_mocktrade,
    'table_creation': test_table_creation,
    'data_write': test_data_write,
    'data_read' : test_data_read,
}