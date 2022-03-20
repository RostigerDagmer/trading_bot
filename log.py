import logging
from datetime import datetime
from tables import *
import os

def initLogger():
    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)

    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H-%M-%S")

    fh = logging.FileHandler(filename=f'log_{now}.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fmt = '%(asctime)s %(filename)s %(levelname)s %(message)s'
    fmt_date = '%Y-%M-%D %H:%M:%S'
    formatter = logging.Formatter(fmt, fmt_date)

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
