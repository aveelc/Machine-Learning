"""Based on https://machinelearningmastery.com/machine-learning-in-python-step-by-step/"""
import sys

import pandas as pd


def print_overview(data_frame, file=''):
    if file:
        print('Saving data frame overview to file', file)
        # If file name is provided redirect stdout temporarily to file
        prev_stdout = sys.stdout
        sys.stdout = open(file, 'w')

    # Set output options to not omit any columns and limit print width
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)

    print('## Data frame info:')
    print(data_frame.info())
    print('\n')

    print('## Data frame shape:')
    print(str(data_frame.shape[0]) + ' rows')
    print(str(data_frame.shape[1]) + ' columns')
    print('\n')

    print_columns(data_frame)

    print('## Data head and tail:')
    print(data_frame.head(10))
    print('...')
    print(data_frame.tail(5))
    print('\n')

    print('## Numeric values statistics:')
    # Note that float format is set
    pd.set_option('float_format', '{:f}'.format)
    print(data_frame.describe(include='all'))
    print('\n')

    # Restore previous stdout
    if file:
        sys.stdout.close()
        sys.stdout = prev_stdout


def print_columns(data_frame):
    print('## Data frame columns:')
    for column in data_frame.columns:
        print(column)
    print('\n')


def print_categorical(data_frame, columns=[], file=''):
    """Prints out all values and values counts in columns where type is object (categorical data)
    columns names array can be provided as an argument. When not provided, data frame columns is default."""
    if file:
        print('Saving data frame categorical columns values and counts to file', file)
        # If file name is provided redirect stdout temporarily to file
        prev_stdout = sys.stdout
        sys.stdout = open(file, 'w')

    print('Non-numeric(categorical) columns values and counts:')

    if columns == []:
        columns = data_frame.columns

    for column in columns:
        # Naive solution assuming that all categorical are objects and there
        # are categories at all
        try:
            if data_frame.dtypes[column] == object:
                print(data_frame.groupby(column).size().to_string(), '\n')
        except KeyError:
            print('ERROR: Column', column, 'not found in data frame')

    # Restore previous stdout
    if file:
        sys.stdout.close()
        sys.stdout = prev_stdout


def print_nan_counts(data_frame):
    for column in data_frame.columns:
        nan_sum = data_frame[column].isna().sum()
        if nan_sum > 0:
            print(column, 'nan values sum: ', nan_sum)
