# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:44:26 2020

@author: xzhang
"""
import streamlit as st
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting.forecasting import plot_ys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def train_test_split(df, y, p, plot = True):
    new_df = df
#    p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
    y_train, y_test = temporal_train_test_split(new_df[y],train_size = p)

    if plot == True:
        st.text("Train Shape")
        st.write(y_train.shape)
        #st.write(y_train)
        st.text("Test Shape")
        st.write(y_test.shape)
        #st.write(y_test)
        plot_ys(y_train, y_test, labels=["y_train", "y_test"])
        st.pyplot()
    return y_train, y_test

def train_test_splitX(df, X, y, p, plot = True):
    new_df = df
#    p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
    train, test = temporal_train_test_split(new_df,train_size = p)
    #x_train = 
    x_train = train[X]
    x_test = test[X]
    
    y_train = train[y]
    y_test = test[y]
    if plot == True:
        st.text("Train Shape")
        st.write(y_train.shape)
        st.text("Test Shape")
        st.write(y_test.shape)
        plot_ys(y_train, y_test, labels=["y_train", "y_test"])
        st.pyplot()
    return x_train, x_test, y_train, y_test

def train_test_splitPH(df, X, y, ts, p, plot = True):
    new_df = df
#    p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
    train, test = temporal_train_test_split(new_df,train_size = p)
    #x_train = 
    x_train = train[X]
    x_test = test[X]
    
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    y_train[['ds','y']] = train[[ts,y]]
    y_test[['ds','y']]  = test[[ts,y]]
    
    if plot == True:
        st.text("Train Shape")
        st.write(y_train.shape)
        st.text("Test Shape")
        st.write(y_test.shape)
        plot_ys(y_train['y'], y_test['y'], labels=["y_train", "y_test"])
        st.pyplot()
    return x_train, x_test, y_train, y_test

def train_test_splitTF(df, X, y, p, plot = True):
    new_df = df
#    p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
    train, test = temporal_train_test_split(new_df,train_size = p)
    #x_train = 
    x_train = train[X]
    x_test = test[X]
    
    y_train = train[[y]]
    y_test = test[[y]]
    
    if plot == True:
        st.text("Train Shape")
        st.write(y_train.shape)
        st.text("Test Shape")
        st.write(y_test.shape)
        plot_ys(y_train[y], y_test[y], labels=["y_train", "y_test"])
        st.pyplot()
    
    return x_train, x_test, y_train, y_test


def multi_train_test_splits(df, y, n, plot = True):
    new_df = df[y].values
#    n = st.sidebar.slider('Select number of splits',1, 10, 3)
    splits = TimeSeriesSplit(n_splits=n)
    index = 1
    y_train = {}
    y_test = {}
    for train_index, test_index in splits.split(new_df):
        y_train[index] = pd.Series(new_df[train_index].flatten())
        y_test[index] = pd.Series(new_df[test_index].flatten())
        y_test[index].index = y_test[index].index + len(y_train[index])
        if plot == True:
            st.text('Observations: %d' % (len(y_train[index]) + len(y_test[index])))
            st.text('Training Observations: %d' % (len(y_train[index])))
            st.text('Testing Observations: %d' % (len(y_test[index])))
            plot_ys(y_train[index], y_test[index], labels=["y_train", "y_test"])
            st.pyplot()
        index += 1
    return y_train, y_test

def multi_train_test_splitsX(df,X,y, n, plot = True):
    new_df = df
#    n = st.sidebar.slider('Select number of splits',1, 10, 3)
    splits = TimeSeriesSplit(n_splits=n)
    index = 1
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    for train_index, test_index in splits.split(new_df):
        x_train[index] = new_df.iloc[train_index][X]
        x_test[index] = new_df.iloc[test_index][X]
        y_train[index] = pd.Series()
        y_test[index] = pd.Series()
        y_train[index] = new_df[y][train_index]
        y_test[index] = new_df[y][test_index]
        if plot == True:
            st.text('Observations: %d' % (len(y_train[index]) + len(y_test[index])))
            st.text('Training Observations: %d' % (len(y_train[index])))
            st.text('Testing Observations: %d' % (len(y_test[index])))
            plot_ys(y_train[index], y_test[index], labels=["y_train", "y_test"])
            st.pyplot()
        index += 1
    return x_train, x_test, y_train, y_test

def multi_train_test_splitsPH(df,X,y,ts, n, plot = True):
    new_df = df
#    n = st.sidebar.slider('Select number of splits',1, 10, 3)
    splits = TimeSeriesSplit(n_splits=n)
    index = 1
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    for train_index, test_index in splits.split(new_df):
        x_train[index] = new_df.iloc[train_index][X]
        x_test[index] = new_df.iloc[test_index][X]
        y_train[index] = pd.DataFrame()
        y_test[index] = pd.DataFrame()
        y_train[index][['ds','y']] = new_df.iloc[train_index][[ts,y]]
        y_test[index][['ds','y']] = new_df.iloc[test_index][[ts,y]]
        if plot == True:
            st.text('Observations: %d' % (len(y_train[index]) + len(y_test[index])))
            st.text('Training Observations: %d' % (len(y_train[index])))
            st.text('Testing Observations: %d' % (len(y_test[index])))
            plot_ys(y_train[index].y, y_test[index].y, labels=["y_train", "y_test"])
            st.pyplot()
        index += 1
    return x_train, x_test, y_train, y_test

def multi_train_test_splitsTF(df,X,y, n, plot = True):
    new_df = df
#    n = st.sidebar.slider('Select number of splits',1, 10, 3)
    splits = TimeSeriesSplit(n_splits=n)
    index = 1
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    for train_index, test_index in splits.split(new_df):
        x_train[index] = new_df.iloc[train_index][X]
        x_test[index] = new_df.iloc[test_index][X]
        y_train[index] = pd.DataFrame()
        y_test[index] = pd.DataFrame()
        y_train[index] = new_df.iloc[train_index][[y]]
        y_test[index] = new_df.iloc[test_index][[y]]
        if plot == True:
            st.text('Observations: %d' % (len(y_train[index]) + len(y_test[index])))
            st.text('Training Observations: %d' % (len(y_train[index])))
            st.text('Testing Observations: %d' % (len(y_test[index])))
            plot_ys(y_train[index][y], y_test[index][y], labels=["y_train", "y_test"])
            st.pyplot()
        index += 1
    return x_train, x_test, y_train, y_test

def walk_forward_validation(df, y, n, window_type):
    new_df = df[y]
#    window_type = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
#    n = st.sidebar.slider('Select number of min observations',int(len(new_df)*0.5), int(len(new_df)*0.95), int(len(new_df)*0.5),step = int(len(new_df)*0.01))
    y_train = {}
    y_test = {}
    index = 1
    if window_type == "Expanding":
        for i in range(n,len(new_df)):
            y_train[index] = new_df[:i]
            y_test[index] = new_df[i:i+1]
            index += 1
    elif window_type == "Sliding":
        for i in range(n,len(new_df)):
            y_train[index] = new_df[i-n:i]
            y_test[index] = new_df[i:i+1]
            index += 1
    #st.text(y_train)     
    #st.text(y_test)
    return y_train, y_test

def walk_forward_validationX(df, X, y, n, window_type):
    new_df = df
#    window_type = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
#    n = st.sidebar.slider('Select number of min observations',int(len(new_df)*0.5), int(len(new_df)*0.98), int(len(new_df)*0.5),step = int(len(new_df)*0.01))
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    index = 1
    if window_type == "Expanding":
        for i in range(n,len(new_df)):
            x_train[index] = new_df.iloc[:i][X]
            x_test[index] = new_df.iloc[i:i+1][X]
            y_train[index] = new_df[y].iloc[:i]
            y_test[index] = new_df[y].iloc[i:i+1]
            index += 1
    elif window_type == "Sliding":
        for i in range(n,len(new_df)):
            x_train[index] = new_df.iloc[i-n:i][X]
            x_test[index] = new_df.iloc[i:i+1][X]
            y_train[index] = pd.DataFrame()
            y_test[index] = pd.DataFrame()
            y_train[index] = new_df[y][i-n:i]
            y_test[index] = new_df[y][i:i+1]
            index += 1
    #st.text(y_train)     
    #st.text(y_test)
    return x_train, x_test, y_train, y_test


def walk_forward_validationPH(df, X, y, ts, n, window_type):
    new_df = df
#    window_type = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
#    n = st.sidebar.slider('Select number of min observations',int(len(new_df)*0.5), int(len(new_df)*0.98), int(len(new_df)*0.5),step = int(len(new_df)*0.01))
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    index = 1
    if window_type == "Expanding":
        for i in range(n,len(new_df)):
            x_train[index] = new_df.iloc[:i][X]
            x_test[index] = new_df.iloc[i:i+1][X]
            y_train[index] = pd.DataFrame()
            y_test[index] = pd.DataFrame()
            y_train[index][['ds','y']] = new_df.iloc[:i][[ts,y]]
            y_test[index][['ds','y']] = new_df.iloc[i:i+1][[ts,y]]
            index += 1
    elif window_type == "Sliding":
        for i in range(n,len(new_df)):
            x_train[index] = new_df.iloc[i-n:i][X]
            x_test[index] = new_df.iloc[i:i+1][X]
            y_train[index] = pd.DataFrame()
            y_test[index] = pd.DataFrame()
            y_train[index][['ds','y']] = new_df[i-n:i][[ts,y]]
            y_test[index][['ds','y']] = new_df[i:i+1][[ts,y]]
            index += 1
    #st.text(y_train)     
    #st.text(y_test)
    return x_train, x_test, y_train, y_test

def walk_forward_validationTF(df, X, y, n, window_type):
    new_df = df
#    window_type = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
#    n = st.sidebar.slider('Select number of min observations',int(len(new_df)*0.5), int(len(new_df)*0.98), int(len(new_df)*0.5),step = int(len(new_df)*0.01))
  
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    index = 1
    if window_type == "Expanding":
        for i in range(n,len(new_df)):
            x_train[index] = new_df.iloc[:i][X]
            x_test[index] = new_df.iloc[i:i+1][X]
            y_train[index] = new_df.iloc[:i][[y]]
            y_test[index] = new_df.iloc[i:i+1][[y]]
            index += 1
    elif window_type == "Sliding":
        for i in range(n,len(new_df)):
            x_train[index] = new_df.iloc[i-n:i][X]
            x_test[index] = new_df.iloc[i:i+1][X]
            y_train[index] = pd.DataFrame()
            y_test[index] = pd.DataFrame()
            y_train[index] = new_df.iloc[i-n:i][[y]]
            y_test[index] = new_df.iloc[i:i+1][[y]]
            index += 1
    #st.text(y_train)     
    #st.text(y_test)
    return x_train, x_test, y_train, y_test