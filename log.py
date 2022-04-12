import logging
from datetime import datetime
from tables import *
import os
import numpy as np
from func import absoluteFilePaths

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


class Database():

    def __init__(self):
        self.datadir = self.find_data_dir()
        self.creation_date = datetime.now()
        print(self.datadir)
        self.file = open_file(os.path.join(self.datadir, "db" + self.creation_date.strftime("%Y_%m_%d_%H-%M-%S") + ".h5"), mode="w", title="db")

    def create_table(self, group, name, t):
        if not self.file.__contains__("/" + group):
            self.file.create_group("/", group, group + " data")
        self.file.create_table("/" + group, name, t)

    def write_data(self, group, table, data):
        t = self.file.get_node("/" + group + "/" + table)
        row = t.row
        for k,v in data.items():
            if isinstance(v, np.ndarray):
                for i in v:
                    row[k] = i
                    row.append()
            else:
                row[k] = v
                row.append()

    def sync(self):
        self.file.flush()

    def find_data_dir(self):
        cwd = os.getcwd()
        files = os.listdir(cwd)
        if not "tmp" in files:
            logging.error("tmp directory not found")
            logging.info("creating tmp directory...")
            os.mkdir(os.path.join(cwd, "tmp"))
        data_path = os.path.join(cwd, "tmp")
        return data_path

    def show(self):
        print(self.file)