import streamlit as st
import pandas as pd
#import sys
import warnings
import seaborn as sns
import time
import numpy as np
from sktime.performance_metrics.forecasting import smape_loss
warnings.filterwarnings('ignore')
#sys.path.append('P:\Python\PythonCode\MA BI Reserving\Mike\Streamlit\time_series_ultils')
#from time_series_plt import * 
#st.title('Time Series - ML')
import time_series_ultils.time_series_plt as tsp
import time_series_ultils.time_series_forecasting as tsf
import time_series_ultils.time_series_split as tss
import time_series_ultils.time_series_utils as tsu

def main():
    st.sidebar.title("What to do")
    activities = ["Exploratory Data Analysis", "Plotting and Visualization", "Building Model", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
   
    # cloud logo
    #st.sidebar.title("Built on:")
    #st.sidebar.image("src/ibmcloud_logo.png", width = 200)
    # Upload file
    st.set_option('deprecation.showfileUploaderEncoding', False)
    #import io
    #file_buffer = st.file_uploader("Choose a CSV file", type="csv")
    #uploaded_file = io.TextIOWrapper(file_buffer)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv","pkl"])

    if uploaded_file is not None and choice == "Exploratory Data Analysis":
        # Add a slider to the sidebar:
        st.sidebar.markdown("# Lag")
#        x = st.sidebar.slider(
#            'Select a lag for ACF and PACF analysis',
#            40, 100
#        )
        # Add a slider to the sidebar:
#        st.sidebar.markdown("# Seasonal")
#        s = st.sidebar.slider(
#            'Select a seasonal parameter from previous ACF and PACF analysis',
#            24, 48
#        )
    
        data = pd.read_csv(uploaded_file)
        st.subheader(choice)
        # Show dataset
        if st.checkbox("Show Dataset"):
            rows = st.number_input("Number of rows", 5, len(data))
            st.dataframe(data.head(rows))
        # Show columns
        if st.checkbox("Columns"):
            st.write(data.columns)
        # Data types
        if st.checkbox("Column types"):
            st.write(tsu.types(data))
        # Show Shape
        if st.checkbox("Shape of Dataset"):
            data_dim = st.radio("Show by", ("Rows", "Columns", "Shape"))
            if data_dim == "Columns":
                st.text("Number of Columns: ")
                st.write(data.shape[1])
            elif data_dim == "Rows":
                st.text("Number of Rows: ")
                st.write(data.shape[0])
            else:
                st.write(data.shape)
        # Check null values in dataset
        if st.checkbox("Check null values"):
            nvalues = tsu.null_values(data)
            st.write(nvalues)
        # Show Data summary
        if st.checkbox("Show Data Summary"):
            st.text("Datatypes Summary")
            st.write(data.describe())
        # Plot time series, ACF and PACF
        if st.checkbox("Select columns for time series"):
            columns = data.columns.tolist()
            selected_ts = st.selectbox("Choose timestamp", columns, index = 0)
            selected_y = st.selectbox("Choose Y", columns, index = 1)
            series = data[selected_y]
            series.index = pd.to_datetime(data[selected_ts], infer_datetime_format = True)
            st.dataframe(series)
            if st.button('Plot Time Series, ACF and PACF'):
                st.write("Deprecated")
#                8/31/2020 plot don't show and run forever
#                tsp.tsplot(series, lags=x)
#                st.pyplot()


    elif uploaded_file is not None and choice == "Plotting and Visualization":
        st.subheader(choice)
        data = pd.read_csv(uploaded_file)
        df = data.copy()
        all_columns = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "line", "scatter", "pie", "bar", "correlation", "distribution"]) 
        
        if type_of_plot=="line":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.line_chart(cust_data)
        
        elif type_of_plot=="area":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.area_chart(cust_data)  
        
        elif type_of_plot=="bar":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.bar_chart(cust_data)
        
        elif type_of_plot=="pie":
            select_columns_to_plot = st.selectbox("Select a column", all_columns)
            st.write(df[select_columns_to_plot].value_counts().plot.pie())
            st.pyplot()
        
        elif type_of_plot=="correlation":
            st.write(sns.heatmap(df.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}))
            st.pyplot()

        elif type_of_plot=="scatter":
            st.write("Scatter Plot")
            scatter_x = st.selectbox("Select a column for X Axis", all_columns)
            scatter_y = st.selectbox("Select a column for Y Axis", all_columns)
            st.write(sns.scatterplot(x=scatter_x, y=scatter_y, data = df))
            st.pyplot()

        elif type_of_plot=="distribution":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            st.write(sns.distplot(df[select_columns_to_plot]))
            st.pyplot()

    elif uploaded_file is not None and choice == "Building Model":
        
        st.subheader(choice)
        data = pd.read_csv(uploaded_file)
        df = data.copy()
        st.write("Select the variables to use for training")
        columns = df.columns.tolist()
        selected_ts = st.selectbox("Choose the timestamp", columns, index = 0)
        selected_column = st.selectbox("Select the target variable", columns, index = 1)
        new_df = df[[selected_ts, selected_column]]
        #new_df.index = pd.to_datetime(data[selected_ts], infer_datetime_format = True)
        st.write(new_df)
            
        model_selection = st.selectbox("Select the model to train", ["Please select", "Auto Pilot - Try all", "AutoArima", "AutoSARIMAX", "Prophet",  "LSTM", "GRU"])
        valid_selection = st.selectbox("Select the validation approach", ["Please select", "Train-Test Split", "Multiple Train-Test Splits", "Walk-Forward Validation"])
        
        if model_selection == "AutoArima":
            
            s = st.sidebar.slider('Select the number of periods in each season',1, 48)
            if valid_selection == "Train-Test Split":
                
                p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
                y_train, y_test = tss.train_test_split(new_df, selected_column, p)
                
                if st.button("Training a Model"):
                    modeling_state = st.text('Training models...')
                    model, y_pred = tsf.forecasting_autoarima(y_train, y_test, s)
                    #st.text(model.summary())
                    tsp.plot_act_pred(y_train, y_test, y_pred)
                    modeling_state.text('Training models...Done!')
                    st.text('All done!!! Symmetric mean absolute percentage error: ' + str(smape_loss(y_test, y_pred)))
             
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred,y_test, model_selection, valid_selection)
            elif valid_selection == "Multiple Train-Test Splits":
                n = st.sidebar.slider('Select number of splits', 2, 10, 3)
                y_train, y_test = tss.multi_train_test_splits(new_df, selected_column, n)
                
                if st.button("Training a Model"):
                    #modeling_state = st.text('Training models...')
                    model = {}
                    y_pred = {}
                    metrics = []
                    for i in range(1,len(y_train)+1):
                        modeling_state = st.text('Training #'+ str(i) + ' models...')
                        model[i], y_pred[i] = tsf.forecasting_autoarima(y_train[i], y_test[i], s)
                        tsp.plot_act_pred(y_train[i], y_test[i], y_pred[i])
                        metrics.append(smape_loss(y_test[i], y_pred[i]))
                        st.text('Symmetric mean absolute percentage error: ' + str(metrics[-1]))
                        # Output model prediction to csv
                        tsu.down_load_csv(df,selected_ts,y_pred,y_test, model_selection, valid_selection)
                        modeling_state.text('Training #'+ str(i) + ' models...Done!')      
                    st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(np.mean(metrics)))
            
            elif valid_selection =="Walk-Forward Validation":
                
                w = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
                n = st.sidebar.slider('Select number of min observations',int(len(new_df)*0.5), int(len(new_df)*0.95), int(len(new_df)*0.5),step = int(len(new_df)*0.01))
                y_train, y_test = tss.walk_forward_validation(new_df, selected_column, n, w)
                
                if len(y_train) != 0:
                    st.text('In total ' + str(len(y_train)) + ' fits')
                if st.button("Training a Model"):
                    model = {}
                    y_pred = pd.Series()
                    metrics = []
                    modeling_state = st.text('Training models... 0% Done!')
                    for i in range(1,len(y_train)+1):    
                        #modeling_state.text('Training models... ' + str(i-1) + '% Done!')
                        model[i], y_pred_tmp = tsf.forecasting_autoarima(y_train[i], y_test[i], s)
                        #st.write(y_pred_tmp)
                        y_pred = y_pred.append(pd.Series(y_pred_tmp))
                        modeling_state.text('Training models... ' + str(round((i)/len(y_train)*100-0.5)) + '% (' + str(i) + '/' + str(len(y_train)) + ') Done!')
                    #st.write(y_pred)
                    #st.write(new_df[len(y_train[1]):])
                    y_pred.index = new_df[len(y_train[1]):].index
                    tsp.plot_act_pred(y_train[1], new_df[len(y_train[1]):][selected_column], y_pred)
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred,new_df[len(y_train[1]):][selected_column], model_selection, valid_selection)

                    st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(smape_loss(new_df[len(y_train[1]):][selected_column], y_pred)))
                
        elif model_selection == "AutoSARIMAX":
            
            s = st.sidebar.slider('Select the number of periods in each season',1, 48)
            selected_X = st.multiselect("Select exogenous variables... leave it blank if no exogenous variables need to include", 
                                        [x for x in columns if x not in selected_column + selected_ts])
            if valid_selection == "Train-Test Split":
                p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
                x_train, x_test, y_train, y_test = tss.train_test_splitX(df, selected_X, selected_column, p)
                
                if st.button("Training a Model"):
                    modeling_state = st.text('Training models...')
                    model, y_pred = tsf.forecasting_autosarimax(x_train, x_test, y_train, y_test, s)
                    st.text(model.summary())
                    tsp.plot_act_pred(y_train, y_test, y_pred)
                    modeling_state.text('Training models...Done!')
                    st.text('All done!!! Symmetric mean absolute percentage error: ' + str(smape_loss(y_test, y_pred)))
             
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred,y_test, model_selection, valid_selection)
                    
            elif valid_selection == "Multiple Train-Test Splits":
                
                n = st.sidebar.slider('Select number of splits', 2, 10, 3)
                x_train, x_test, y_train, y_test = tss.multi_train_test_splitsX(df, selected_X, selected_column, n)
                
                if st.button("Training a Model"):
                    #modeling_state = st.text('Training models...')
                    model = {}
                    y_pred = {}
                    metrics = []
                    for i in range(1,len(y_train)+1):
                        modeling_state = st.text('Training #'+ str(i) + ' models...')
                        model[i], y_pred[i] = tsf.forecasting_autosarimax(x_train[i], x_test[i], y_train[i], y_test[i], s)
                        tsp.plot_act_pred(y_train[i], y_test[i], y_pred[i])
                        metrics.append(smape_loss(y_test[i], y_pred[i]))
                        st.text('Symmetric mean absolute percentage error: ' + str(metrics[-1]))
                        # Output model prediction to csv
                        tsu.down_load_csv(df,selected_ts,y_pred,y_test, model_selection, valid_selection)
                        modeling_state.text('Training #'+ str(i) + ' models...Done!')
                    st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(np.mean(metrics)))
                    
            elif valid_selection =="Walk-Forward Validation":
                
                w = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
                n = st.sidebar.slider('Select number of min observations',int(len(df)*0.5), int(len(df)*0.95), int(len(df)*0.5),step = int(len(df)*0.01))
                x_train, x_test, y_train, y_test = tss.walk_forward_validationX(df, selected_X, selected_column, n, w)
                
                if len(y_train) != 0:
                    st.text('In total ' + str(len(y_train)) + ' fits')
                if st.button("Training a Model"):
                    model = {}
                    y_pred = pd.Series()
                    metrics = []
                    modeling_state = st.text('Training models... 0% Done!')
                    for i in range(1,len(y_train)+1):
                        #modeling_state.text('Training models... ' + str(i-1) + '% Done!')
                        model[i], y_pred_tmp = tsf.forecasting_autosarimax(x_train[i], x_test[i], y_train[i], y_test[i], s)
                        #st.write(y_pred_tmp)
                        y_pred = y_pred.append(pd.Series(y_pred_tmp))
                        modeling_state.text('Training models... ' + str(round((i)/len(y_train)*100-0.5)) + '% (' + str(i) + '/' + str(len(y_train)) + ') Done!')
                    #st.write(y_pred)
                    #st.write(new_df[len(y_train[1]):])
                    y_pred.index = df.iloc[len(y_train[1]):].index
                    tsp.plot_act_pred(y_train[1], df.iloc[len(y_train[1]):][selected_column], y_pred)
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred,df.iloc[len(y_train[1]):][selected_column], model_selection, valid_selection)

                    st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(smape_loss(df.iloc[len(y_train[1]):][selected_column], y_pred)))
             
                # Output model prediction to csv
                   #down_load_csv(df,selected_ts,y_pred,y_test)
                   
                   
        elif model_selection == "Prophet":
            
            selected_X = st.multiselect("Select additional regressors... leave it blank if no additional variables need to include",
                                        [x for x in columns if x not in selected_column + selected_ts])
            if valid_selection == "Train-Test Split":
                
                p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
                x_train, x_test, y_train, y_test = tss.train_test_splitPH(df,selected_X,selected_column,selected_ts, p)
                
                if st.button("Training a Model"):
                    modeling_state = st.text('Training models...')
                    model, y_pred = tsf.forecasting_prophet(x_train, x_test, y_train, y_test)
                    tsp.plot_act_pred(y_train.y, y_test.y, y_pred.yhat)              
                    modeling_state.text('Training models...Done!')
                    st.text('All done!!! Symmetric mean absolute percentage error: ' + str(smape_loss(y_test.y, y_pred.yhat)))
             
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred.yhat,y_test.y, model_selection, valid_selection)
                    
                    
            elif valid_selection == "Multiple Train-Test Splits":
                
                n = st.sidebar.slider('Select number of splits', 2, 10, 3)
                x_train, x_test, y_train, y_test = tss.multi_train_test_splitsPH(df,selected_X,selected_column,selected_ts, n)
                            
                if st.button("Training a Model"):
                #modeling_state = st.text('Training models...')
                    model = {}
                    y_pred = {}
                    metrics = []
                    for i in range(1,len(y_train)+1):
                        modeling_state = st.text('Training #'+ str(i) + ' models...')
                        model[i], y_pred[i] = tsf.forecasting_prophet(x_train[i], x_test[i], y_train[i], y_test[i])
                        tsp.plot_act_pred(y_train[i].y, y_test[i].y, y_pred[i].yhat)
                        metrics.append(smape_loss(y_test[i].y, y_pred[i].yhat))
                        st.text('Symmetric mean absolute percentage error: ' + str(metrics[-1]))
                                    # Output model prediction to csv
                        tsu.down_load_csv(df,selected_ts,y_pred[i].yhat,y_test[i].y, model_selection, valid_selection)
                        modeling_state.text('Training #'+ str(i) + ' models...Done!')      
                        st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(np.mean(metrics)))    
            
            elif valid_selection == "Walk-Forward Validation":
                
                w = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
                n = st.sidebar.slider('Select number of min observations',int(len(df)*0.5), int(len(df)*0.95), int(len(df)*0.5),step = int(len(df)*0.01))
                x_train, x_test, y_train, y_test = tss.walk_forward_validationPH(df, selected_X, selected_column, selected_ts, n, w)
                
                if len(y_train) != 0:
                    st.text('In total ' + str(len(y_train)) + ' fits')
                if st.button("Training a Model"):
                    model = {}
                    y_pred = pd.Series()
                    metrics = []
                    modeling_state = st.text('Training models... 0% Done!')
                    for i in range(1,len(y_train)+1):
                        #modeling_state.text('Training models... ' + str(i-1) + '% Done!')
                        model[i], y_pred_tmp = tsf.forecasting_prophet(x_train[i], x_test[i], y_train[i], y_test[i])
                        #st.write(y_pred_tmp)
                        y_pred = y_pred.append(pd.Series(y_pred_tmp.yhat))
                        modeling_state.text('Training models... ' + str(round((i)/len(y_train)*100-0.5)) + '% (' + str(i) + '/' + str(len(y_train)) + ') Done!')
                    #st.write(y_pred)
                    #st.write(new_df[len(y_train[1]):])
                    y_pred.index = df.iloc[len(y_train[1]):].index
                    tsp.plot_act_pred(y_train[1].y, df.iloc[len(y_train[1]):][selected_column], y_pred)
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred,df.iloc[len(y_train[1]):][selected_column], model_selection, valid_selection)

                    st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(smape_loss(df.iloc[len(y_train[1]):][selected_column], y_pred)))

        elif model_selection in ("LSTM", "GRU"):
            
            selected_X = st.multiselect("Select additional regressors... leave it blank if no additional variables need to include",
                                        [x for x in columns if x not in selected_column + selected_ts])
            n_ep = st.sidebar.slider('Select number of epochs',100, 1000, 300, step = 50)
            n_timestep = st.sidebar.slider('Select number of time steps',1, round(len(df)/4), 12, step = 1)
            if valid_selection == "Train-Test Split":
                p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
                x_train, x_test, y_train, y_test = tss.train_test_splitTF(df,selected_X,selected_column, p)         
                if st.button("Training a Model"):
                    modeling_state = st.text('Training models...')
                    model, hist, y_pred = tsf.forecasting_RNN(x_train, x_test, y_train, y_test, n_ep, n_timestep, model_selection)
                    tsp.plot_hist_loss(hist)
                    tsp.plot_act_pred(y_train[selected_column], y_test[selected_column], y_pred)              
                    modeling_state.text('Training models...Done!')
                    st.text('All done!!! Symmetric mean absolute percentage error: ' + str(smape_loss(y_test[selected_column], y_pred)))
             
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred,y_test[selected_column], model_selection, valid_selection)
                    
            elif valid_selection == "Multiple Train-Test Splits":
                
                n = st.sidebar.slider('Select number of splits', 2, 10, 3)
                x_train, x_test, y_train, y_test = tss.multi_train_test_splitsTF(df, selected_X, selected_column, n)
                if st.button("Training a Model"):
                #modeling_state = st.text('Training models...')
                    model = {}
                    y_pred = {}
                    hist = {}
                    metrics = []
                    for i in range(1,len(y_train)+1):
                        modeling_state = st.text('Training #'+ str(i) + ' models...')
                        model[i], hist[i], y_pred[i] = tsf.forecasting_RNN(x_train[i], x_test[i], y_train[i], y_test[i], n_ep, n_timestep, model_selection)
                        tsp.plot_hist_loss(hist[i])
                        
                        tsp.plot_act_pred(y_train[i][selected_column], y_test[i][selected_column], y_pred[i])
                        metrics.append(smape_loss(y_test[i][selected_column], y_pred[i]))
                        st.text('Symmetric mean absolute percentage error: ' + str(metrics[-1]))
                                    # Output model prediction to csv
                        tsu.down_load_csv(df,selected_ts,y_pred[i],y_test[i][selected_column], model_selection, valid_selection)
                        modeling_state.text('Training #'+ str(i) + ' models...Done!')      
                        st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(np.mean(metrics)))
                        
            elif valid_selection =="Walk-Forward Validation":
                
                w = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
                n = st.sidebar.slider('Select number of min observations',int(len(df)*0.5), int(len(df)*0.95), int(len(df)*0.5),step = int(len(df)*0.01))
                x_train, x_test, y_train, y_test = tss.walk_forward_validationTF(df, selected_X, selected_column, n, w)
                
                if len(y_train) != 0:
                    st.text('In total ' + str(len(y_train)) + ' fits')
                if st.button("Training a Model"):
                    model = {}
                    hist = {}
                    y_pred = pd.Series()
                    metrics = []
                    modeling_state = st.text('Training models... 0% Done!')
                    for i in range(1,len(y_train)+1):
                        #modeling_state.text('Training models... ' + str(i-1) + '% Done!')
                        model[i], hist[i], y_pred_tmp = tsf.forecasting_RNN(x_train[i], x_test[i], y_train[i], y_test[i], n_ep, n_timestep, model_selection)
                        #st.write(y_pred_tmp)
                        y_pred = y_pred.append(pd.Series(y_pred_tmp))
                        modeling_state.text('Training models... ' + str(round((i)/len(y_train)*100-0.5)) + '% (' + str(i) + '/' + str(len(y_train)) + ') Done!')
                    #st.write(y_pred)
                    #st.write(new_df[len(y_train[1]):])
                    y_pred.index = df.iloc[len(y_train[1]):].index
                    tsp.plot_act_pred(y_train[1][selected_column], df.iloc[len(y_train[1]):][selected_column], y_pred)
                    # Output model prediction to csv
                    tsu.down_load_csv(df,selected_ts,y_pred,df.iloc[len(y_train[1]):][selected_column], model_selection, valid_selection)

                    st.text('All done!!! Average symmetric mean absolute percentage error: ' + str(smape_loss(df.iloc[len(y_train[1]):][selected_column], y_pred)))
                    
                    
        elif model_selection == "Auto Pilot - Try all":
            #st.text("The app will try all possible algorithms and take more time...")
            s = st.sidebar.slider('Select the number of periods in each season',1, 48, 12)
            selected_X = st.multiselect("Select exogenous variables... leave it blank if no exogenous variables need to include", 
                                        [x for x in columns if x not in selected_column + selected_ts])
            model_candidate = ["AutoArima", "AutoSARIMAX", "Prophet",  "LSTM", "GRU"]
            
            if valid_selection == "Train-Test Split":
                p = st.sidebar.slider('Select the percentage of training',0, 100, 75)/100
                if st.button("Start piloting"):
                    modeling_state = st.text('Training models...')
                    model_candidate_i = 0
                    loss = pd.DataFrame(columns = ('Model','loss','runtime(sec)'))
                    for alg in model_candidate:
                        if alg == "AutoArima":
                            start_time = time.time()
                            y_train, y_test = tss.train_test_split(new_df, selected_column, p, plot = False)
                            model, y_pred = tsf.forecasting_autoarima(y_train, y_test, s, verbose = False)
                            tsp.plot_act_pred(y_train, y_test, y_pred)
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(y_test, y_pred), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                        elif alg == "AutoSARIMAX":
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.train_test_splitX(df, selected_X, selected_column, p, plot = False)
                            model, y_pred = tsf.forecasting_autosarimax(x_train, x_test, y_train, y_test, s, verbose = False)
                            tsp.plot_act_pred(y_train, y_test, y_pred)
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(y_test, y_pred), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                        elif alg == "Prophet":
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.train_test_splitPH(df,selected_X,selected_column,selected_ts, p, plot = False)
                            model, y_pred = tsf.forecasting_prophet(x_train, x_test, y_train, y_test)
                            tsp.plot_act_pred(y_train.y, y_test.y, y_pred.yhat)  
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(y_test.y, y_pred.yhat), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                        elif alg in ("LSTM", "GRU"):
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.train_test_splitTF(df,selected_X,selected_column, p, plot = False)
                            n_ep = 300
                            n_timestep = s
                            model, hist, y_pred = tsf.forecasting_RNN(x_train, x_test, y_train, y_test, n_ep, n_timestep, alg)
                            tsp.plot_act_pred(y_train[selected_column], y_test[selected_column], y_pred)  
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(y_test[selected_column], y_pred), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                    st.text('All done!!! Best symmetric mean absolute percentage error: ' + str(round(loss.loss.min(),4)) + ' ' + loss[loss.loss == loss.loss.min()].iat[0,0])
                    st.write(loss)
            
            if valid_selection == "Multiple Train-Test Splits":
                n = st.sidebar.slider('Select number of splits', 2, 10, 3)
                if st.button("Start piloting"):
                    modeling_state = st.text('Training models...')
                    model_candidate_i = 0
                    loss = pd.DataFrame(columns = ('Model','loss','runtime(sec)'))
                    for alg in model_candidate:
                        
                        if alg == "AutoArima":
                            start_time = time.time()
                            y_train, y_test = tss.multi_train_test_splits(new_df, selected_column, n, plot = False)
                            model = {}
                            y_pred = {}
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                model[i], y_pred[i] = tsf.forecasting_autoarima(y_train[i], y_test[i], s, verbose = False)
                                tsp.plot_act_pred(y_train[i], y_test[i], y_pred[i])
                                metrics.append(smape_loss(y_test[i], y_pred[i]))    

                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, np.mean(metrics), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                        elif alg == "AutoSARIMAX":
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.multi_train_test_splitsX(df, selected_X, selected_column, n, plot = False)
                            model = {}
                            y_pred = {}
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                    model[i], y_pred[i] = tsf.forecasting_autosarimax(x_train[i], x_test[i], y_train[i], y_test[i], s, verbose = False)
                                    tsp.plot_act_pred(y_train[i], y_test[i], y_pred[i])
                                    metrics.append(smape_loss(y_test[i], y_pred[i]))
        
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, np.mean(metrics), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')                   
                            
                        elif alg == "Prophet":
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.multi_train_test_splitsPH(df,selected_X,selected_column,selected_ts, n, plot = False)
                            model = {}
                            y_pred = {}
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                    model[i], y_pred[i] = tsf.forecasting_prophet(x_train[i], x_test[i], y_train[i], y_test[i])
                                    tsp.plot_act_pred(y_train[i].y, y_test[i].y, y_pred[i].yhat)
                                    metrics.append(smape_loss(y_test[i].y, y_pred[i].yhat))
        
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, np.mean(metrics), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                                        
                        elif alg in ("LSTM", "GRU"):
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.multi_train_test_splitsTF(df, selected_X, selected_column, n, plot = False)
                            n_ep = 300
                            n_timestep = s
                            model = {}
                            y_pred = {}
                            hist = {}
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                    model[i], hist[i], y_pred[i] = tsf.forecasting_RNN(x_train[i], x_test[i], y_train[i], y_test[i], n_ep, n_timestep, alg)
                                    tsp.plot_act_pred(y_train[i][selected_column], y_test[i][selected_column], y_pred[i])
                                    metrics.append(smape_loss(y_test[i][selected_column], y_pred[i]))
        
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, np.mean(metrics), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                    st.text('All done!!! Best symmetric mean absolute percentage error: ' + str(round(loss.loss.min(),4)) + ' ' + loss[loss.loss == loss.loss.min()].iat[0,0])
                    st.write(loss)
                    
            if valid_selection == "Walk-Forward Validation":
                
                w = st.selectbox("Window type", ["Please select", "Expanding", "Sliding"])
                n = st.sidebar.slider('Select number of min observations',int(len(df)*0.5), int(len(df)*0.98), int(len(df)*0.5),step = int(len(df)*0.01))
                st.text('In total ' + str(len(df) - n) + ' fits for each algorithm')
                
                if st.button("Start piloting"):
                    
                    modeling_state = st.text('Training models...')
                    model_candidate_i = 0
                    loss = pd.DataFrame(columns = ('Model','loss','runtime(sec)'))
                    
                    for alg in model_candidate:
                        
                        if alg == "AutoArima":
                
                            start_time = time.time()
                            y_train, y_test = tss.walk_forward_validation(new_df, selected_column, n, w)
                            model = {}
                            y_pred = pd.Series()
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                model[i], y_pred_tmp = tsf.forecasting_autoarima(y_train[i], y_test[i], s, verbose = False)
                                y_pred = y_pred.append(pd.Series(y_pred_tmp))
                            
                            y_pred.index = new_df[len(y_train[1]):].index
                            tsp.plot_act_pred(y_train[1], new_df[len(y_train[1]):][selected_column], y_pred)    

                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(new_df[len(y_train[1]):][selected_column], y_pred), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                        elif alg == "AutoSARIMAX":
                            
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.walk_forward_validationX(df, selected_X, selected_column, n, w)
                            model = {}
                            y_pred = pd.Series()
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                model[i], y_pred_tmp = tsf.forecasting_autosarimax(x_train[i], x_test[i], y_train[i], y_test[i], s, verbose = False)
                                y_pred = y_pred.append(pd.Series(y_pred_tmp))
                            
                            y_pred.index = df.iloc[len(y_train[1]):].index
                            tsp.plot_act_pred(y_train[1], df.iloc[len(y_train[1]):][selected_column], y_pred)
        
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(df.iloc[len(y_train[1]):][selected_column], y_pred), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')      
                
                        elif alg == "Prophet":
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.walk_forward_validationPH(df, selected_X, selected_column, selected_ts, n, w)
                            model = {}
                            y_pred = pd.Series()
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                model[i], y_pred_tmp = tsf.forecasting_prophet(x_train[i], x_test[i], y_train[i], y_test[i])
                                y_pred = y_pred.append(pd.Series(y_pred_tmp.yhat))
        
                            y_pred.index = df.iloc[len(y_train[1]):].index
                            tsp.plot_act_pred(y_train[1].y, df.iloc[len(y_train[1]):][selected_column], y_pred)
                            
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(df.iloc[len(y_train[1]):][selected_column], y_pred), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')

                        elif alg in ("LSTM", "GRU"):
                            start_time = time.time()
                            x_train, x_test, y_train, y_test = tss.walk_forward_validationTF(df, selected_X, selected_column, n, w)
                            n_ep = 300
                            n_timestep = s
                            model = {}
                            hist = {}
                            y_pred = pd.Series()
                            metrics = []
                            for i in range(1,len(y_train)+1):
                                
                                model[i], hist[i], y_pred_tmp = tsf.forecasting_RNN(x_train[i], x_test[i], y_train[i], y_test[i], n_ep, n_timestep, alg)
                                y_pred = y_pred.append(pd.Series(y_pred_tmp))
                            
                            y_pred.index = df.iloc[len(y_train[1]):].index
                            tsp.plot_act_pred(y_train[1][selected_column], df.iloc[len(y_train[1]):][selected_column], y_pred)
        
                            st.text(alg + ' done!!!')
                            run_time = round(time.time() - start_time,2)
                            loss.loc[model_candidate_i] = [alg, smape_loss(df.iloc[len(y_train[1]):][selected_column], y_pred), run_time]
                            model_candidate_i += 1
                            modeling_state.text('Training models...'  + str(model_candidate_i) + '/' + str(len(model_candidate)) + ' Done!')
                            
                    st.text('All done!!! Best symmetric mean absolute percentage error: ' + str(round(loss.loss.min(),4)) + ' ' + loss[loss.loss == loss.loss.min()].iat[0,0])
                    st.write(loss)

                    
    elif choice == "About":
        st.title("About")
        st.write("Modeling part was implemented by Mike Z")
        st.write("EDA and Visual Ref: https://github.com/Alro10/streamlit-time-series")
        st.write("Source code: P:\Python\PythonCode\AAQuickML")

if __name__ == "__main__":
    main()