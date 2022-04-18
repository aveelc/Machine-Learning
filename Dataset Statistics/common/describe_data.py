"""Based on https://machinelearningmastery.com/machine-learning-in-python-step-by-step/"""
import pandas as pd


def print_overview(data_frame):
    """ Print pandas data frame overview"""
    print('## Data frame info:')
    print(data_frame.info())
    print('\n')

    print('## Data frame shape:')
    print(str(data_frame.shape[0]) + ' rows')
    print(str(data_frame.shape[1]) + ' columns')
    print('\n')

    print('## Data frame columns:')
    for column in data_frame.columns:
        print(column)
    print('\n')

    print('## Data head and tail:')
    print(data_frame.head(10))
    print('...')
    print(data_frame.tail(5))
    print('\n')

    print('Numeric values statistics:')
    # Note that float format is set
    pd.set_option('float_format', '{:f}'.format)
    print(data_frame.describe())
    print('\n')
