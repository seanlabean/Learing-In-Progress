import collections
import os
import time
import yt
yt.funcs.mylog.setLevel(40)
import argparse
import multiprocessing
import glob
from functools import partial

def load_plt_file(input_file):
    ds = yt.load(input_file)
    return ds

def locate_sinks(data_structure):
    ad = ds.all_data()
    sinks = ad['particle_type'] > 1.0

    sinks[]
    return
def report_gas_data_from_sink():
    
    return
