import pandas as pd
from ayb import ayb


def main(param2val):
    """This function is run by Ludwig on remote workers."""

    performance = ayb(param2val)
    print(performance)

    series_list = []
    for k, v in performance.items():
        s = pd.Series(v, index=k)
        s.name = 'took'
        series_list.append(s)
    return series_list
