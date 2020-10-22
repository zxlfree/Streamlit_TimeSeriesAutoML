# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:44:39 2020

@author: xzhang
"""
import streamlit as st
from sktime.forecasting.arima import AutoARIMA
import numpy as np
from pmdarima import auto_arima
import pandas as pd
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def forecasting_autoarima(y_train, y_test, s, verbose = True):
    fh = np.arange(len(y_test)) + 1
    forecaster = AutoARIMA(sp=s)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    if verbose == True:
        st.text(forecaster.get_fitted_params())
    #st.text(forecaster.get_params())
    return forecaster, y_pred

def forecasting_autosarimax(x_train, x_test, y_train, y_test, s, verbose = True):
    #fh = np.arange(len(y_test)) + 1
    if x_train.empty == True:
        arima = auto_arima(y = y_train, m=s, trace = True)
        y_pred = pd.Series(arima.predict(n_periods = len(x_test)))
    else:
        arima = auto_arima(y = y_train, exogenous = x_train, m=s, trace = True)
        y_pred = pd.Series(arima.predict(n_periods = len(x_test), exogenous = x_test))
    y_pred.index = y_test.index
    if verbose == True:
        st.text(arima.get_params())
    return arima, y_pred

def forecasting_prophet(x_train, x_test, y_train, y_test):
    
    m = Prophet()
    for col in x_train.columns:
        m.add_regressor(col)
    m.fit(pd.concat([y_train,x_train], axis = 1))
    forecast = m.predict(pd.concat([y_test[['ds']],x_test], axis = 1))
    forecast.index = y_test.index
    return m, forecast[['ds', 'yhat']]

def multi_step_pred_TF(model, X_test, steps):
    for i in range(steps):
        if i == 0:
            y_pred = model.predict(X_test[0:1])
        else:
            X_test[i,-1,-1] = y_pred[-1, 0]
            y_pred = np.append(y_pred, model.predict(X_test[i:i+1]),axis = 0)
    return y_pred

def create_dataset(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        #print(i)
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def forecasting_RNN(x_train, x_test, y_train, y_test, n_ep, n_timestep, model_type):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    scaled_x_train = x_train.copy()
    scaled_x_test = x_test.copy()
    scaled_y_train = y_train.copy()
    scaled_y_test = y_test.copy()
    
    if x_train.empty == False:
        scaled_x_train[:] = x_scaler.fit_transform(x_train[:])
        scaled_x_test[:] = x_scaler.transform(x_test[:])

    scaled_y_train[:] = y_scaler.fit_transform(y_train[:])
    scaled_y_test[:] = y_scaler.transform(y_test[:])

    time_steps = n_timestep

    X_train, Y_train = create_dataset(pd.concat([scaled_x_train,scaled_y_train], axis = 1), scaled_y_train, time_steps)
    X_test, Y_test = create_dataset(pd.concat([pd.concat([scaled_x_train.iloc[-time_steps:],scaled_x_test]),
                                               pd.concat([scaled_y_train.iloc[-time_steps:],scaled_y_test])], axis = 1), 
                                pd.concat([scaled_y_train.iloc[-time_steps:],scaled_y_test]), time_steps)

    model = keras.Sequential()
    earlystrop_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 40)

    if model_type == "LSTM":
        model.add(keras.layers.LSTM(
          units=128,
          #input_shape=(X_train.shape[1], X_train.shape[2])
        ))
    elif model_type == "GRU":
        model.add(keras.layers.GRU(
          units=128,
          #input_shape=(X_train.shape[1], X_train.shape[2])
        ))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(units=1))
    model.compile(
      loss='mean_squared_error',
      optimizer=keras.optimizers.Adam(0.001)
    )
    
    history = model.fit(
        X_train, Y_train,
        epochs=n_ep,
        batch_size=16,
        validation_split=0.2,
        #validation_data = (X_test, Y_test),
        verbose=0,
        shuffle=False,
        callbacks = [earlystrop_callback]
    )
    
#    plt.plot(history.history['loss'], label='train')
#    plt.plot(history.history['val_loss'], label='test')
#    plt.legend()
#    st.pyplot()
    
    y_pred = multi_step_pred_TF(model, X_test, len(X_test))
    y_pred = pd.Series(y_scaler.inverse_transform(y_pred).flatten())
    y_pred.index = np.arange(len(y_train), len(y_train) + len(Y_test))
    
    return model, history, y_pred