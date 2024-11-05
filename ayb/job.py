import pandas as pd
from .ayb import run_ayb
import argparse

def main(param2val):
    """This function is run by Ludwig on remote workers."""
    location = "ludwig_local"

    performance = run_ayb(param2val, location)
    print(performance)

    series_list = []
    for k, v in performance.items():
        s = pd.Series(v, index=k)
        s.name = 'took'
        series_list.append(s)
    return series_list
