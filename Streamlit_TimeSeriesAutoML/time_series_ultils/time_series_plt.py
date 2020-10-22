# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:44:00 2020

@author: xzhang
"""
import matplotlib.pyplot as plt
from sktime.utils.plotting.forecasting import plot_ys
import statsmodels.tsa.api as smt
import streamlit as st
#import sys
#print(sys.path)

def tsplot(y, lags=None, figsize=(20, 12), style='bmh'):
    """
        Plot time series, its ACF and PACF
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
        
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        

def plot_act_pred(y_train, y_test, y_pred):
    plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    st.pyplot()

def plot_hist_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    st.pyplot()        