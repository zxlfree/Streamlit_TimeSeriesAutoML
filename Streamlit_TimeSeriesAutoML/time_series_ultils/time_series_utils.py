# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:45:44 2020

@author: xzhang
"""
import streamlit as st
import pandas as pd
import base64


def null_values(df):
    null_test = (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False).index
    null_data_test = pd.concat([
        df.isnull().sum(axis = 0),
        (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False),
        df.loc[:, df.columns.isin(list(null_test))].dtypes], axis=1)
    null_data_test = null_data_test.rename(columns={0: '# null', 
                                        1: '% null', 
                                        2: 'type'}).sort_values(ascending=False, by = '% null')
    null_data_test = null_data_test[null_data_test["# null"]!=0]
    
    return null_data_test

def types(df):
    return pd.DataFrame(df.dtypes, columns=['Type'])

def down_load_csv(df,selected_ts,y_pred,y_test, model, valid):
    output = pd.DataFrame([df[selected_ts].tolist()[-len(y_pred):],y_pred, y_test]).transpose()
    output.columns = [selected_ts, 'pred', 'actual']
    csv_exp = output.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download= "{model}_{valid}_prediction.csv">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)