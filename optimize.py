import numpy as np
from skopt import forest_minimize
from skopt.optimizer import dummy_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from test import mocktrade, vec_mocktrade
import random
from pyswarms.single.global_best import GlobalBestPSO
from multiprocessing.pool import ThreadPool as Pool
from api import *


# default ~ 42 days
def forest_search_on_history(history_length=60000):
    search_space = [Integer(1, 15, name="timestep"), Integer(1, 100, name="w_l"), Integer(1, 100, name="w_h")]
    n_tries = 10
    api = MockAPI(datapath='data/Bitstamp_ETHUSD_2021_minute.csv', readlines=613813)

    @use_named_args(search_space)
    def objective(**params):
        timeframes = np.linspace(500, 613813, n_tries, dtype=int)
        params_list = [{**params, 'api': api, 'timeframe': f} for f in timeframes]
        results = np.array([-1.0 * float(mocktrade(**p)) for p in params_list])
        return np.mean(results)

    def step(result):
        print(f'current best:\ntimestep={result.x[0]}, w_l={result.x[1]}, w_h={result.x[2]} --> profit: {-1.0 * result.fun}')
        print(f'params: timestep={result.x_iters[-1][0]}, w_l={result.x_iters[-1][1]}, w_h={result.x_iters[-1][2]}')
        print(f'profit: {-1.0 * result.func_vals[-1]}')

    results = forest_minimize(objective, search_space, callback=step, n_jobs=-1)
    return results

def swarm_on_history():
    n_tries = 20
    def objective(p):
        params_list = [{'timestep': int(parm[0]), 'w_l': int(parm[1]), 'w_h': int(parm[2]), 'readlines': 613813} for parm in p]
        return np.array([-1.0 * float(mocktrade(**p)) for p in params_list])

    p_min = np.ones(3)
    p_max = np.ones(3) * 5000
    bounds = (p_min, p_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=32, dimensions=3, options=options, bounds=bounds)
    profit, params = optimizer.optimize(objective, iters=10)
    print(profit)
    return params


BLOCK_DIM = 16

def vec_forest_search_on_history():
    search_space = [Integer(1, 15, name="timestep"), Integer(2, 100, name="w_l"), Integer(2, 100, name="w_h")]
    api = MockAPI(datapath='data/Bitstamp_ETHUSD_2021_minute.csv', readlines=613813)
    history = api.close_history_float()
    lows = api.history_lows(1)
    highs = api.history_highs(1)

    st = 613813 // BLOCK_DIM
    print(st)

    with Pool(BLOCK_DIM) as threadpool:

        @use_named_args(search_space)
        def objective(**params):
            #timeframes = np.linspace(500, 613813, BLOCK_DIM, dtype=int)
            timeframe_start = [st * t for t in range(BLOCK_DIM)]
            timeframe_end = [st * (t + 1) for t in range(BLOCK_DIM)]
            timesteps = np.repeat(params['timestep'], BLOCK_DIM)
            windows_low = np.repeat(params['w_l'], BLOCK_DIM)
            windows_high = np.repeat(params['w_h'], BLOCK_DIM)
            results = vec_mocktrade(timesteps, windows_low, windows_high, timeframe_end, timeframe_start, lows=lows, highs=highs, history=history, threadpool=threadpool)
            return -1.0 * np.mean(np.array(results))

        def step(result):
            print(f'current best:\ntimestep={result.x[0]}, w_l={result.x[1]}, w_h={result.x[2]} --> profit: {-1.0 * result.fun}')
            print(f'params: timestep={result.x_iters[-1][0]}, w_l={result.x_iters[-1][1]}, w_h={result.x_iters[-1][2]}')
            print(f'profit: {-1.0 * result.func_vals[-1]}')

        results = forest_minimize(objective, search_space, callback=step, n_jobs=1, n_calls=1000)
        return results


def exaustive_search_on_history():
    search_space = [Integer(1, 15, name="timestep"), Integer(2, 100, name="w_l"), Integer(2, 100, name="w_h")]
    api = MockAPI(datapath='data/Bitstamp_ETHUSD_2021_minute.csv', readlines=613813)
    history = api.close_history_float()
    lows = api.history_lows(1)
    highs = api.history_highs(1)

    st = 613813 // BLOCK_DIM

    with Pool(BLOCK_DIM) as threadpool:

        @use_named_args(search_space)
        def objective(**params):
            #timeframes = np.linspace(500, 613813, BLOCK_DIM, dtype=int)
            timeframe_start = [st * t for t in range(BLOCK_DIM)]
            timeframe_end = [st * (t + 1) for t in range(BLOCK_DIM)]
            timesteps = np.repeat(params['timestep'], BLOCK_DIM)
            windows_low = np.repeat(params['w_l'], BLOCK_DIM)
            windows_high = np.repeat(params['w_h'], BLOCK_DIM)
            results = vec_mocktrade(timesteps, windows_low, windows_high, timeframe_end, timeframe_start, lows=lows, highs=highs, history=history, threadpool=threadpool)
            return -1.0 * np.mean(np.array(results))

        def step(result):
            print(f'current best:\ntimestep={result.x[0]}, w_l={result.x[1]}, w_h={result.x[2]} --> profit: {-1.0 * result.fun}')
            print(f'params: timestep={result.x_iters[-1][0]}, w_l={result.x_iters[-1][1]}, w_h={result.x_iters[-1][2]}')
            print(f'profit: {-1.0 * result.func_vals[-1]}')

        results = dummy_minimize(objective, search_space, callback=step, initial_point_generator="grid", n_calls=(15 * 99 * 99))
        return result
