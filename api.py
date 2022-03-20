from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import numpy as np
from abc import ABC, abstractmethod
import logging
import csv
import time
import ciso8601
from datetime import datetime, timedelta
import numpy as np
from decimal import Decimal
import pandas as pd
from ftx_client import FtxClient
from func import all_equal
import pytz
from dateutil import tz
import logging

class APIResponse(ABC):
    def __init__(self, data):
        self.data = data
        super().__init__()
    
    @abstractmethod
    def numpy(self) -> np.array:
        pass
    
    @abstractmethod
    def json(self) -> json:
        pass
    
    @abstractmethod
    def text(self) -> str:
        pass
    


class API(ABC):
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.data = None
        self.session = Session()
    
    @property
    def headers(self):
        return {
            'Accepts' : 'application/json',
            'X-CMC_PRO_API_KEY': self.key
        }

    @abstractmethod
    def __call__(self, symbol, time):
        pass

    @abstractmethod
    def get_price(self) -> APIResponse:
        pass


    def log_response(self, response):
        if response.status_code != 200:
            logging.warn(response.status_code)
            logging.debug(response.text)
            return
        logging.debug(f'{response.status_code}: {response.text}')
    
    def __call__(self, t):
        pass

    def get_price(self):
        pass

    def history_length(self):
        pass
    
    def open_history(self, minute_interval=1, timeframe=None):
        pass
    
    def open_history_float(self,  minute_interval=1, timeframe=None):
        pass

    def close_history(self, minute_interval=1, timeframe=None):
        pass

    def close_history_float(self, minute_interval=1, timeframe=None):
        pass

    def history_highs(self, minute_interval, timeframe=None):
        pass
    
    def history_lows(self, minute_interval, timeframe=None):
        pass


class CMarketCapResponse(APIResponse):

    def numpy(self):
        pass

    def json(self):
        return json.loads(self.data.text)

    def text(self):
        return self.data.text


class CMarketCapAPI(API):

    def __call__(self, symbol, time):
        self.session.headers.update(self.headers)
        param = {
            #symbol: str(symbol)
            'symbol': symbol,
            'convert': 'USD',
        }
        try:
            self.data = self.session.get(self.url, params=param)

        except (ConnectionError, Timeout, TooManyRedirects) as e:
            logging.warn(e)

    def get_price(self):
        print(self.data)
        return CMarketCapResponse(self.data)


class FtxResponse(APIResponse):

    def numpy(self):
        pass

    def json(self):
        return json.loads(self.data.text)

    def text(self):
        return self.data.text

'''
    Wrapper around FTX client that caches history locally
'''
class FtxAPI(API):

    def __init__(self, api_key, api_secret, subaccount_name=None, currency='ETH'):
        self.client = FtxClient(api_key, api_secret, subaccount_name)
        self.history = {}
        '''
            history = {
                'currency_symbol': {
                    'low'  : np.ndarray, # lows as float64
                    'high' : np.ndarray, # highs as float64
                    'open'  : np.ndarray, # open prices as float64
                    'close' : np.ndarray, # close prices as float64
                    'date' : np.ndarray, # timestamps as strings
                    'orders': np.ndarray, # holds order IDS
                }
            }
        '''
        self.currency = currency + '/USD'
        self.funds = Decimal(self.client.get_account_info()['freeCollateral'])
        self.time_format = "%Y-%M-%DT%h:%m:%s%z"
        self.timezone = tz.tzlocal()
        self.next_update = 60.0
    
    def log_stats(self):
            logging.info(f'time:   {self.history[self.currency]["date"][-1]}')
            logging.info('current stats:')
            logging.info(f'low:    {self.history[self.currency]["low"][-1]}$')
            logging.info(f'high:   {self.history[self.currency]["high"][-1]}$')
            logging.info(f'open:   {self.history[self.currency]["open"][-1]}$')
            logging.info(f'close:  {self.history[self.currency]["close"][-1]}$')

    def update(self, verbose=True):

        end_time = datetime.now()
        start_time = self.last_history_update().astimezone(self.timezone)

        if (end_time.replace(tzinfo=self.timezone) - start_time).total_seconds() < 60.0:
            logging.info('-- everything still up to date --')
            if verbose:
                self.log_stats()

        response_array = self.client.get_historical_prices( self.currency, 
                                                            resolution=60, 
                                                            start_time=time.mktime(start_time.timetuple()), 
                                                            end_time=time.mktime(end_time.timetuple()))
        
        if response_array[-1]['startTime'] == self.history[self.currency]['date'][-1]:
            logging.info('no new info yet')

        for k in ['low', 'high', 'open', 'close']:
            self.history[self.currency][k] = np.append(self.history[self.currency][k], self._wrap_dict_ndarray(response_array, k, float)[-1])
        
        self.history[self.currency]['date'] = np.append(self.history[self.currency]['date'], self._wrap_dict_ndarray(response_array, 'startTime', str)[-1])
        if verbose:
            logging.info('-------- updated history ---------')
            self.log_stats()
        

    def auto_update(self, verbose=False):
        while True:
            offset = timedelta(seconds=3)
            now = datetime.now()
            dt = timedelta(minutes=0, seconds=now.second, microseconds=now.microsecond)
            last_minute = now - dt
            next_minute = last_minute + timedelta(minutes=1)
            dw = next_minute - datetime.now() + offset
            self.next_update = next_minute
            time.sleep(dw.total_seconds())
            self.update(verbose)
    
    def _wrap_type(self, a, t):
        if isinstance(a, np.ndarray):
            return np.array([t(v) for v in a])
        return t(a)
    
    def _wrap_dict_ndarray(self, a, k, t):
        return np.array([t(v[k]) for v in a])

    # the timestep parameter is there only to get the correct length of history
    # the history will always have a frequency of 1 minute
    def init_history(self, **kwargs):
        length = kwargs['length']
        timestep = kwargs['timestep']
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=length * timestep)

        # array of dict
        response_array = self.client.get_historical_prices( self.currency, 
                                                            resolution=60, 
                                                            start_time=time.mktime(start_time.timetuple()), 
                                                            end_time=time.mktime(end_time.timetuple()))

        self.history[self.currency] = {}
        for k in ['low', 'high', 'open', 'close']:
            self.history[self.currency][k] = self._wrap_dict_ndarray(response_array, k, float)
        
        self.history[self.currency]['date'] = self._wrap_dict_ndarray(response_array, 'startTime', str)

    def history_length(self):
        keys = ['low', 'high', 'open', 'close']
        lengths = [self.history[self.currency][key].shape[0] for key in keys]
        if not all_equal(lengths):
            logging.error('theres something wrong with the history\n lengths are out of sync')
            for l,k in zip(lengths, keys):
                logging.info(f'length of {k}s: {l}')
        return lengths[0]
    
    def last_history_update(self):
        return datetime.fromisoformat(self.history[self.currency]['date'][-1])

    def open_history(self, minute_interval=1, timeframe=None):
        return self._wrap_decimal(self.history[self.currency]['open'][0:timeframe:minute_interval])
    
    def open_history_float(self,  minute_interval=1, timeframe=None):
        return self.history[self.currency]['open'][0:timeframe:minute_interval]

    def close_history(self, minute_interval=1, timeframe=None):
        return self._wrap_decimal(self.history[self.currency]['close'][0:timeframe:minute_interval])

    def close_history_float(self, minute_interval=1, timeframe=None):
        return self.history[self.currency]['close'][0:timeframe:minute_interval]

    def history_highs(self, minute_interval, timeframe=None):
        history_high = self.history[self.currency]['high']
        return np.amax(history_high[:((timeframe or history_high.shape[0]) // minute_interval) * minute_interval].reshape([(timeframe or history_high.shape[0]) // minute_interval, minute_interval]), axis=1)
    
    def history_lows(self, minute_interval, timeframe=None):
        history_low = self.history[self.currency]['low']
        return np.amin(history_low[:((timeframe or history_low.shape[0]) // minute_interval) * minute_interval].reshape([(timeframe or history_low.shape[0]) // minute_interval, minute_interval]), axis=1)

    def get_orderbook(self, depth):
        return self.client.get_orderbook(self.currency, depth)

'''
////////////// MOCK CLASSES ////////////////
'''

def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
 
    while low <= high:

        mid = (high + low) // 2
 
        # If x is greater, ignore left half
        if arr[mid] > x:
            low = mid + 1
 
        # If x is smaller, ignore right half
        elif arr[mid] < x:
            high = mid - 1
 
        # means x is present at mid
        else:
            return mid
    
    if low == 0:
        return -1
    # If we reach here, then the element was not present
    if low > len(arr) - 1:
        return -2
    
    else:
        # we want to return the index that holds the value closest to the query
        l_dist = abs(arr[low] - x)
        r_dist = abs(arr[high] - x)
        return low if l_dist < r_dist else high

class MockAPI(API):

    def __init__(self, datapath, readlines=100):
        self.datapath = datapath
        self.data = {}
        with open(datapath) as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            header = next(reader)
            self.data = dict((el, []) for el in header)
            for i in range(readlines):
                l = next(reader)
                for j, key in enumerate(header):
                    self.data[key].append(l[j])

        convert_date = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        self.data['unix'] = np.flip(np.array([int(stamp) for stamp in self.data['unix']]))
        self.data['date'] = np.flip(np.array(self.data['date']))
        self.data['Volume ETH'] = np.flip(np.array([float(v) for v in self.data['Volume ETH']]))
        self.data['Volume USD'] = np.flip(np.array([float(v) for v in self.data['Volume USD']]))
        self.data['open'] = np.flip(np.array([float(p) for p in self.data['open']]))
        self.data['close'] = np.flip(np.array([float(p) for p in self.data['close']]))
        self.history_low = np.flip(np.array([float(e) for e in self.data['low']]))
        self.history_high = np.flip(np.array([float(e) for e in self.data['high']]))
    	
    def data_frame(self, minute_interval=None):
        date = self.data['date'][0::minute_interval][:-1]
        ope = self.open_history_float(minute_interval)[:-1]
        close = self.close_history_float(minute_interval)[:-1]
        high = self.history_highs(minute_interval)
        low = self.history_lows(minute_interval)

        print(date.shape)
        print(ope.shape)
        print(close.shape)
        print(high.shape)
        print(low.shape)

        df = pd.DataFrame({
            'date': date,
            'open': ope,
            'close': close,
            'high': high,
            'low': low,
        })
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')))
        return df

    def __call__(self, t):
        ts = ciso8601.parse_datetime(t)
        index = binary_search(self.data['unix'], int(time.mktime(ts.timetuple())))

    def get_price(self):
        pass

    def history_length(self):
        return len(self.data['unix'])
    
    def open_history(self, minute_interval=1, timeframe=None):
        return np.array([Decimal(v) for v in self.data['open'][0:timeframe:minute_interval]])
    
    def open_history_float(self,  minute_interval=1, timeframe=None):
        return self.data['open'][0:timeframe:minute_interval]

    def close_history(self, minute_interval=1, timeframe=None):
        return np.array([Decimal(v) for v in self.data['close'][0:timeframe:minute_interval]])

    def close_history_float(self, minute_interval=1, timeframe=None):
        return self.data['close'][0:timeframe:minute_interval]

    def history_highs(self, minute_interval, timeframe=None):
        return np.amax(self.history_high[:((timeframe or self.history_high.shape[0]) // minute_interval) * minute_interval].reshape([(timeframe or self.history_high.shape[0]) // minute_interval, minute_interval]), axis=1)
    
    def history_lows(self, minute_interval, timeframe=None):
        return np.amin(self.history_low[:((timeframe or self.history_low.shape[0]) // minute_interval) * minute_interval].reshape([(timeframe or self.history_low.shape[0]) // minute_interval, minute_interval]), axis=1)
    