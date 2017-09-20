
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from datetime import datetime, timedelta
import warnings
from sklearn_pandas import DataFrameMapper
import logging
from tqdm import tqdm
import random
from functools import reduce
from operator import add
warnings.filterwarnings("ignore")

def load_file(file):
    '''
    
    load file into dataframe
    
    '''
    data = pd.read_csv(file, sep = ',')
    data = data.iloc[2:]
    columns = list(data.columns)
    drops = ['Currency','Next_Day_Open_Price_Movement','Open_Price_movement','High_Price_movement','Low_Price_movement','Close_Price_movement']
    dropcols = [column for column in drops if column in columns]
    if dropcols:
        dataset_input = data.drop(dropcols, axis = 1)
    return dataset_input

def na_handling(data, arg):
    '''
    
    select whether to delete, interpolation to handle rows with NAs
    
    '''
    if arg == 1:
        data_nona = data.dropna()
        if (len(data_nona.index) == len(data.index)):
            pass
        else:
            print 'Rows: From ' + str(len(data.index)) + ' to ' + str(len(data_nona.index))
    elif arg == 2:
        data_nona = data.interpolate()
        # if time series, method='quadratic'
        # if values approximating a cumulative distribution function, method = 'pchip'
        # if goal of smooth plotting, method='akima'
        data_nona = data_nona.dropna() #drop all rows that were not imputed
        print 'Remaining decreased to ' + str(len(data_nona.index)) + ' from ' + str(len(data.index))
    else:
        print 'arg = 1 or 2'
        print 'call function again'
    return data_nona

def preprocess_cat(dataset_input):
    dataset_input = remove_inc_variables(dataset_input, 0.1)
    dataset_input = na_handling(dataset_input, 1)
    datecol = ['Date']
    categories =  list(dataset_input.select_dtypes(include=['O']).columns)
    numbers = list(dataset_input.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns)

    mapper = DataFrameMapper(
            [('Date', None)] +
            [(category, preprocessing.LabelEncoder()) for category in categories] +
            [(number, None) for number in numbers], df_out = True)
    
    transformedData = mapper.fit_transform(dataset_input)
    dummified = pd.get_dummies(transformedData, columns = categories)
    #print list(dummified.columns)
    drops = []
    for i in categories:
        dummy_i = [col for col in dummified.columns if i in col]
        #print "Dummify Variable - ",i, dummy_i
        #drop the 1st of the list
        drops.append(dummy_i[0])
    #print drops
    dataset_input2 = dummified.drop(drops, axis = 1)
    return dataset_input2, mapper, numbers

def preprocess_num(dataset_input, num_cols):

    numbers = num_cols
    categories = [col for col in list(dataset_input.columns) if col not in numbers][1:]
    means_dict = {}
    std_dict = {}
    
    #save means and std of fit
    for i in numbers:
        means_dict[i] = dataset_input.ix[:,i].mean()
        std_dict[i] = dataset_input.ix[:,i].std(ddof=0)
    
    mapper = DataFrameMapper(
            [('Date', None)] +
            [(category, None) for category in categories] +
            [(number, preprocessing.StandardScaler()) for number in numbers], df_out = True)
    
    transformedData = mapper.fit_transform(dataset_input)
    return transformedData, mapper, means_dict, std_dict

def remove_inc_variables(data, pct):
    '''
    
    Check Missing Values - col wise, then row-wise
    if missing values per column is greater than 10% of total row count, remove
    
    '''
    col_to_keep = []
    starting = len(data.columns)
    for i in range(len(data.columns)):
        if (float(data.iloc[:,i].isnull().sum().tolist()) / float(len(data.index))) > pct:
            print str(data.iloc[:,i].name) + " Removing | NAs: " + str(round(float(data.iloc[:,i].isnull().sum().tolist()) / float(len(data.index)),2))
        else:
            col_to_keep.append(i)
    data_nona = data.iloc[:,col_to_keep]
    ending = len(data_nona.columns)
    print "Variables: From %d to %d" % (starting, ending)

def load_data(dataset, seq_len, num_cols):
    '''
    
    load data properly into a keras model. it has to be a 3D array.
    
    '''
    
    amount_of_features = len(dataset.columns) - 1 #(remove dates)
    print '1. Counting num of features...', amount_of_features
    
    print '2. Separating training and testing data...'
    row = int(0.9 * len(dataset.index))
    train = dataset.iloc[:row,:]
    test = dataset.iloc[row:,:]
    print '   > Train rows:', len(train.index)
    print '   > Test rows:', len(test.index)
    
    print '3. Fitting & transforming mapper_num to training data...'          
    train, mapper_num, means_dict, std_dict = preprocess_num(train, num_cols)
    allcols = list(train.columns)
    
    print '4. Transforming test data'
    test = mapper_num.transform(test)
    
    train_data =  train.iloc[:,1:].as_matrix()
    train_date_col = train.ix[:,'Date'].as_matrix()
         
    test_data =  test.iloc[:,1:].as_matrix()
    test_date_col = test.ix[:,'Date'].as_matrix()
        
    sequence_length = seq_len + 1
    train_result = []
    train_dates = []
    test_result = []
    test_dates = []
    
    for index in range(len(train_data) - sequence_length):
        seq = train_data[index: index + sequence_length]
        date_seq = train_date_col[index: index + sequence_length]

        train_result.append(seq)
        train_dates.append(date_seq)
    
    
    train_result = np.array(train_result)
    print '   > TRAIN:', train_result.shape
    
    for index in range(len(test_data) - sequence_length):
        seq = test_data[index: index + sequence_length]
        date_seq = test_date_col[index: index + sequence_length]
        test_result.append(seq)
        test_dates.append(date_seq)
    
    
    test_result = np.array(test_result)
    print '   > TEST:', test_result.shape
    
    print '5. Shuffling training set...'
    np.random.shuffle(train_result)
    
    print '6. Splitting Xs & Ys...'
    
    x_train = train_result[:, : -1]
    print '   > x_train:', x_train.shape
    y_train = train_result[:,-1][:,19:23]
    print '   > y_train:', y_train.shape
    y_test_dates = test_dates
    x_test = test_result[:, : -1]
    print '   > x_test:', x_test.shape
    y_test = test_result[:, -1][:,19:23]
    print '   > y_test:', y_test.shape
    
    print '7. Reshaping for model input... x_train:', [x_train.shape[0], x_train.shape[1], amount_of_features], 'x_test:',  [x_test.shape[0], x_test.shape[1], amount_of_features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
    
    print '8. Done!'
    
    return [x_train, y_train, x_test, y_test, y_test_dates, mapper_num, means_dict, std_dict, allcols]


    return data_nona