import numpy as np
from decimal import Decimal
import time
import threading
from func import *
import logging
from dateutil import tz
from datetime import datetime
import os
import sys

################## archival trading functions ##################

def preprocess(signals, fee, api=None, price_history=None):
    if api:
        price_history = api.close_history()
    go_long = signals[:,0]
    end_long = signals[:, 1]
    go_short = signals[:,1] + 1
    end_short = signals[:, 0] - 1

    # boundary checks
    end_long[-1] = min(len(price_history) - 2, end_long[-1])
    end_short[-1] = min(len(price_history) - 2, end_short[-1])
    go_long[-1] = min(len(price_history) - 2, go_long[-1])
    go_short[-1] = min(len(price_history) - 2, go_short[-1])

    randoms = np.random.sample([4, signals.shape[0]]).astype(price_history.dtype)
    if isinstance(price_history[0], Decimal):
        randoms = np.array([Decimal(s) for axis in randoms for s in axis]).reshape(4, signals.shape[0])
    sample = lambda x, d, s: x + (d * s)

    price_long_opened = sample(price_history[go_long], price_history[go_long] - price_history[go_long + 1], randoms[0])
    price_long_closed = sample(price_history[end_long], price_history[end_long] - price_history[end_long + 1], randoms[1])
    price_short_opened = sample(price_history[go_short], price_history[go_short] - price_history[go_short + 1], randoms[2])
    price_short_closed = sample(price_history[end_short], price_history[end_short] - price_history[end_short + 1], randoms[3])
    
    return [price_long_opened, price_long_closed, price_short_opened, price_short_closed]

def run_trades(c, discount, prices, signals):
    price_long_opened = prices[0]
    price_long_closed = prices[1]
    price_short_opened = prices[2]
    price_short_closed = prices[3]  
    c *= (price_long_closed[0] / price_long_opened[0]) * discount
    for I in range(1, len(signals)):
        c *= (price_long_closed[I] / price_long_opened[I]) * discount
        c *= (price_short_opened[I-1] / price_short_closed[I-1]) * discount
    c *= (price_short_opened[len(signals) - 1] / price_short_closed[len(signals) - 1]) * discount
    return c

# run using copy of price history and floating point arithmetic
def static_trade_history(price_history, signals, fee):
    preprocessed = preprocess(signals, fee, price_history=price_history)
    c = 1.0
    discount = 1.0 - fee
    return run_trades(c, discount, preprocessed, signals)

# run using copy of price history and fixed point arithmetic
def dec_static_trade_history(price_history, signals, fee):
    preprocessed = preprocess(signals, fee, price_history=price_history)
    c = Decimal(1.0)
    discount = Decimal(1.0 - fee)
    return run_trades(c, discount, preprocessed, signals)

# run using api object (bad)
def trade_history(api, signals, fee):
    preprocessed = preprocess(signals, fee, api=api)
    c = Decimal(1.0)
    discount = Decimal(1.0 - fee)
    return run_trades(c, discount, preprocessed, signals)


def trade_plot(api, signals, fee, leverage=1.0):
    prices = preprocess(signals, fee, api=api)
    price_long_opened = prices[0]
    price_long_closed = prices[1]
    price_short_opened = prices[2]
    price_short_closed = prices[3]  

    c = Decimal(1.0)
    discount = Decimal(1.0 - fee)
    leverage = Decimal(leverage)
    c *= (price_long_closed[0] / price_long_opened[0]) * leverage * discount
    ax = [c]

    # TODO: check price history between trades if a margin call occurs.
    for I in range(1, len(signals)):
        c *= (price_long_closed[I] / price_long_opened[I]) * leverage * discount
        ax.append(c)
        c *= (price_short_opened[I-1] / price_short_closed[I-1]) * leverage * discount
        ax.append(c)
    c *= (price_short_opened[len(signals) - 1] / price_short_closed[len(signals) - 1]) * leverage * discount
    ax.append(c)
    return (ax, price_long_opened, price_long_closed, price_short_opened, price_short_closed)

def vec_trade_history(close_prices, signals ,n_dim):
    pass

# (1/leverage) (position_open - current_price)

FEE = 0.0007
usleep = lambda x: time.sleep(x/1000000.0)

################## Interactive Bot #####################

class Bot:

    def __init__(self, api, strategies, history=None, db=None):
        self.strategies = strategies
        self.api = api
        self.last_update = self.api.last_history_update()
        self.signals = {} # only for plotting
        self.ema = {}
        self.capital = self.api.funds
        self.open_positions = {}
        self.timezone = tz.tzlocal()
        self.database = db
        self.record_type = np.dtype([("trades", "S16"), 
                    ("ema_low", np.float64), 
                    ("ema_high", np.float64), 
                    ("capital", np.float64)
                    ])

    def fetch_all_as_db_record(self, strategy):
        return {'trades': ['None'] * self.ema[strategy]['low'].shape[0], 
                'ema_low': self.ema[strategy]['low'],
                'ema_high': self.ema[strategy]['high'],
                'captial': [float(self.capital)] * self.ema[strategy]['low'].shape[0]}

    def fetch_latest_as_db_record(self, strategy):
        none_as_str = lambda n: 'None' if not n else str(n)
        return {'trades': none_as_str(open_positions[strategy]),
                'ema_low': self.ema[strategy]['low'][-1],
                'ema_high': self.ema[strategy]['high'][-1],
                'capital': float(self.capital)}

    def init_db(self):
        for strategy in self.strategies:
            self.database.create_table("bot", strategy, self.record_type)
            self.database.write_data("/bot", f'/{strategy}', self.fetch_all_as_db_record(strategy))

    def write_out(self):
        for strategy in self.strategies:
            self.database.write_data('/bot', f'/{strategy}', self.fetch_latest_as_db_record(strategy))
        self.database.sync()

    def init_ema(self):
        for strategy, (timestep, window_low, window_high) in self.strategies.items():
            self.ema[strategy] = {}
            self.ema[strategy]['low'] = ema(self.api.history_lows(timestep), window_low)
            self.ema[strategy]['high'] = ema(self.api.history_highs(timestep), window_high)

    def run_thread(self, strategy, params):
        trade = self.eval(strategy, params)
        logging.info(f'[{strategy}] executing trade: {trade}')
        self.execute(strategy, trade)
        if strategy == 'main':
            self.submit(trade)

    def ema_empty(self):
        return not any([self.ema.get(s) for s in self.strategies]) 

    def run(self, delta=100):
        first = True
        while True:
            tic = time.time()
            usleep(delta)
            #####################################
            logging.info("running bot thread")
            if self.ema_empty():
                self.init_ema()
                if self.database:
                    self.init_db()
            threads = []
            for strategy, params in self.strategies.items():
                t = threading.Thread(target=self.run_thread, args=(strategy, params))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            if self.database and not first:
                self.write_data()
                first = False
            #####################################
            toc = time.time()
            
            now = datetime.now()
            dw = self.api.next_update - now
            time.sleep(dw.total_seconds())


    def execute(self, strategy, trade):
        if strategy != 'main':
            if trade:
                if trade[0] == 'resolve':
                    capital = self.resolve_position(strategy, trade[1])
                    time.sleep(0.2) # wait for submission to api
                    self.open_position(strategy, trade[1], capital)
                elif trade[0] == 'noop':
                    self.open_position(strategy, trade[1], capital)
                else:
                    logging.error(f'error executing {trade} for [{strategy}]')

    def submit(self, trade):
        # TODO: replace this with an actual API call
        if trade:
            if trade[0] == 'resolve':
                capital = self.resolve_position('main', trade[1])
                time.sleep(0.2) # wait for submission to api
                self.open_position('main', trade[1], capital)
            elif trade[0] == 'noop':
                self.open_position('main', trade[1], self.capital)
            else:
                logging.error(f'error submitting {trade} for [{strategy}]')


    def resolve_position(self, strategy, position):
        if self.open_positions.get(strategy):
            holding = self.capital / Decimal(self.open_positions[strategy][1])
            logging.info(f'resolving long position of {holding} {self.api.currency}')
            sale_price = self.price_heuristic(self.capital, 'asks') * (1 - FEE)
            holding_price = self.open_positions[strategy][1] # fee already applied

            logging.info(f'Bought in at: {holding_price} USD')
            logging.info(f'selling at:   {sale_price} USD')
            diff = 0.0

            if position == "golong": # we have to resolve goshort
                diff = holding_price / sale_price
            elif position == "goshort": # we have to revolve golong
                diff = sale_price / holding_price
            else:
                logging.error(f"error resolving position {position}. Unknown")
                return Decimal(0.0)

            # clear open position record
            self.open_positions[strategy] = None
            logging.info(f'new capital: {self.capital * Decimal(diff)}')
            return self.capital * Decimal(diff)

        logging.info('no position to resolve yet...')
        return capital

    def open_position(self, strategy, position, capital):
        if self.open_positions.get(strategy):
            logging.error(f'something went wrong... this position should be closed!\nstrategy: {strategy}\nposition: {self.open_positions[strategy]}')
            return
        logging.info('openining position...')
        # highest volume bid
        maxbid = self.price_heuristic(self.capital, 'bids')
        logging.info(f'opened {position[2:]} position at {maxbid} USD')
        self.open_positions[strategy] = (position, maxbid * (1 - FEE))

    def price_heuristic(self, capital, side):
        # TODO: use capital and orderbook to estimate somewhat realistic sale price
        side = np.array(self.api.get_orderbook(10)[side])
        # highest volume side (ask, bid)
        maxside = side.transpose()[0, side.argmax(axis=0)][-1]
        # TODO: maybe introduce a discount factor here... or something cleverer
        #       maybe proportional to the difference between own bid size
        return maxside


    def eval(self, strategy, params):
        timestep, window_low, window_high = params
        low = self.api.history_lows(timestep)[-1]
        high = self.api.history_highs(timestep)[-1]

        # caculate EMAs for right now
        emal_ = next_ema(self.ema[strategy]['low'], low, window_low)
        emah_ = next_ema(self.ema[strategy]['high'], high, window_low)

        if self.api.history_length() % timestep in [0, 1]:
            self.ema[strategy]['low'] = np.append(self.ema[strategy]['low'], emal_)
            self.ema[strategy]['high'] = np.append(self.ema[strategy]['high'], emah_)
        else:
            self.ema[strategy]['low'][-1] = emal_
            self.ema[strategy]['high'][-1] = emah_
        
        logging.info(f"new_ema_low: {self.ema[strategy]['low'][-5:]}")
        logging.info(f"new_ema_high: {self.ema[strategy]['high'][-5:]}")

        if (self.ema[strategy]['low'][-1] > self.ema[strategy]['high'][-1]) and (self.ema[strategy]['low'][-2] <= self.ema[strategy]['high'][-2]):
            logging.info(f'strategy {strategy} found a signal to go long')
            try: 
                if self.open_positions.get(strategy)[0] == "goshort":
                    return ("resolve", "golong")
            except (NameError, AttributeError, TypeError) as e:
                logging.info(f'no short position to resolve. Going long')
                return ('noop', 'golong')
        
        elif (self.ema[strategy]['low'][-1] < self.ema[strategy]['high'][-1]) and (self.ema[strategy]['low'][-2] >= self.ema[strategy]['high'][-2]):
            logging.info(f'strategy {strategy} found a signal to go short')
            try:
                if self.open_positions.get(strategy)[0] == "golong":
                    return ("resolve", "goshort")
            except (NameError, AttributeError, TypeError) as e:
                logging.info(f'no long position to resolve. Going short')
                return ('noop', 'goshort')

        else:
            return None
