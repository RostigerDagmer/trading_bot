import argparse
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
from api import CMarketCapAPI, CMarketCapResponse, MockAPI, FtxAPI
from test import tests, mocktrade
from func import ema
from matplotlib import pyplot as plt
from optimize import vec_forest_search_on_history, forest_search_on_history, swarm_on_history, exaustive_search_on_history
from func import signals
from bot import trade_plot, Bot
from decimal import Decimal
from plot import *
import threading
from datetime import datetime, timedelta
from dateutil import tz
import time
import pytz
from log import initLogger, Database
import logging
import platform
import signal
import sys

API_KEY = '8e26e098-6e2f-4275-818b-6fc64e2c15a0'
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

initLogger()
LOG = logging.getLogger('root')

def test(params):
    tests_ = tests
    if params:
        tests_ = {k: v for k, v in tests_.items() if k in params}
    for t in tests_:
        separator = 'testing-----------------------------------------------'
        separator = separator[:len(separator) - len(t)] + t + "()"
        print(separator)
        tests[t]()

def optimize():
    #params = vec_forest_search_on_history()
    params = exaustive_search_on_history()
    #params = forest_search_on_history()
    #params = swarm_on_history()
    print(f'Parameters found: {params}')

def plot(params):
    optional_get = lambda d, k, v: v if not d else d.get(k) if d.get(k) else v

    timestep = optional_get(params, 'timestep', 13)
    window_l = optional_get(params, 'window_low', 2)
    window_h = optional_get(params, 'window_high', 24)
    api = MockAPI(datapath='data/Bitstamp_ETHUSD_2021_minute.csv', readlines=613813)
    #animate_mocktrade(api, 13, 2, 24, True)
    ema_plot(api, timestep, window_l, window_h)

def p1():
    load_dotenv(find_dotenv())
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    api = FtxAPI(api_key, api_secret)
    def price_heuristic(capital, side):
        # TODO: use capital and orderbook to estimate somewhat realistic sale price
        side = np.array(api.get_orderbook(10)[side])
        # highest volume side (ask, bid)
        maxside = side.transpose()[0, side.argmax(axis=0)][-1]
        # TODO: maybe introduce a discount factor here... or something cleverer
        #       maybe proportional to the difference between own bid size
        return maxside
    print(price_heuristic(1000.0, 'bids'))

def play_ground_function():
    offset = timedelta(seconds=3)
    now = datetime.now()
    dt = timedelta(minutes=0, seconds=now.second, microseconds=now.microsecond)
    last_minute = now - dt
    next_minute = last_minute + timedelta(minutes=1)

    dw = next_minute - datetime.now() + offset

    print(dt.total_seconds())
    print(last_minute)
    print(next_minute)
    print(dw.total_seconds())



def start_threads(threads):
    for t in threads:
        t.start()

def join_threads(threads):
    for t in threads:
        t.join()

def live_mocktrade(params):
    load_dotenv(find_dotenv())
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')

    threads = []

    optional_get = lambda d, k, v: v if not d else d.get(k) if d.get(k) else v

    timestep = optional_get(params, 'timestep', 13)
    window_l = optional_get(params, 'window_low', 2)
    window_h = optional_get(params, 'window_high', 24)

    db = Database()

    LOG.info('creating API thread...')

    api = FtxAPI(api_key, api_secret)
    api.init_history(timestep=timestep, length=max(window_l, window_h))
    api_thread = threading.Thread(target=api.auto_update, kwargs={'verbose': True})
    api_thread.setDaemon(True)

    strategies = {
        'main': (timestep, window_l, window_h)
        # optional strategies that evaluate but don't trade
    }

    LOG.info('creating Bot thread...')

    bot = Bot(api, strategies, db=db)
    bot_thread = threading.Thread(target=bot.run, kwargs={'delta': 7000000})
    bot_thread.setDaemon(True)

    try:
        api_thread.start()
        bot_thread.start()
        if platform.system() == "Windows":
            os.system("pause")
        elif platform.system() == "Linux":
            signal.pause()
    except (KeyboardInterrupt, SystemExit):
        print('BYEEEEEEEE! BYEBYEYY!')
        api_thread.stop()
        bot_thread.stop()
        sys.exit()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_tests', action="store_true")
    parser.add_argument('-o', action="store_true") 
    parser.add_argument('-f', action="store_true")
    parser.add_argument('-p', action="store_true")
    parser.add_argument('-lm', action="store_true")

    parser.add_argument('--ema_params', nargs='+', help="list of parameters with [timestep, window_low, window_high].\nCan be named i.e. [timestep=12, window_low=4]")
    parser.add_argument('-t', nargs='+', help="list of tests to execute. --list-tests for a list of all avaibale test cases.")
    flags = parser.parse_args()
    params = flags.ema_params
    if params:
        try: 
            params = dict([(p.split('=')[0], int(p.split('=')[1])) for p in params])
        except:
            params = dict(zip(["timestep", "window_low", "window_high"], [int(p) for p in params]))
    if flags.t:
        test(flags.t)
    elif flags.list_tests:
        print(tests.keys())
    elif flags.o:
        optimize()
    elif flags.p:
        plot(params)
    elif flags.lm:
        live_mocktrade(params)
    else:
        play_ground_function(params)


if __name__ == "__main__":
    main()