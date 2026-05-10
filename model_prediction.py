import pickle
import pandas as pd
import numpy as np
from train.data_processing_utils import date_feature_engineering
from model_utils import load_model, load_latest_data, get_best_model

def populate_df_row(og_df):
    # update last row with lag, rolling meand and std attributes
    df = og_df.copy()
    df.loc[len(df)] = np.nan
    for window in [1, 7, 30]:
        lag_col_name = f'lag_{window}'
        rolling_mean_col = f"rolling_mean_{window}"
        rolling_std_col = f"rolling_std_{window}"
        df[lag_col_name] = df['Total'].shift(window)
        df[rolling_mean_col] = df['Total'].transform(lambda x: x.shift(1).rolling(window).mean())
        if window != 1:
            df[rolling_std_col] = df['Total'].transform(lambda x: x.shift(1).rolling(window).std())
        
    # Update date 
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df.loc[len(df)-1, 'Date'] = df.loc[len(df)-2, 'Date'] + pd.Timedelta(weeks=1)
    df.loc[len(df)-1, 'State'] = df.loc[len(df)-2, 'State']
    df = date_feature_engineering(df)
    return df.iloc[-1]


    
def predict_xgb(df, weeks=1):
    for _ in range(weeks):
        df.loc[len(df)] = populate_df_row(df)
        # Extract populated last row for forcasting
        model_input = df.drop(columns=['Date', 'Total']).iloc[-1].values.reshape(1, -1)

        xgb_model = load_model('XGBoost', 'XGBoost')
        output = xgb_model.predict(model_input)

        # Add prediction to the dataframe
        df.loc[len(df)-1, 'Total'] = output[0]
    return df.tail(weeks)



def predict_prophet(state, last_date, weeks=1):
    date_series = pd.DataFrame({
        'ds': pd.date_range(start=last_date, periods=weeks+1, freq='W'),
    })

    prophet_model = load_model('Prophet', f'state_{state}')
    output = prophet_model.predict(date_series)
    return output['yhat'].iloc[:].numpy()


def predict_arima(state, weeks=1):
    arima_model = load_model("ARIMA", f"state_{state}")
    output = arima_model.forecast(steps=weeks)
    return np.array(output)

# print(predict_prophet(35, '2023-05-14', 6))
# print(predict_arima(35, 6))



def handle_xgb_prediction(state, weeks):
    data_versions_list = pd.read_json("Data_Version_History.json")
    latest_filename = data_versions_list['filename'].iloc[-1]
    filepath = f'version_data/{latest_filename}.csv'
    df = pd.read_csv(filepath)

    df = df.groupby('State').get_group(state)[-40:].copy().reset_index()
    df = df.drop(columns=["index", "Unnamed: 0"], errors="ignore")

    result_df = predict_xgb(df, weeks)
    return result_df['Total'].iloc[-weeks:].to_numpy()


def best_model_prediction(state, weeks, l_date):
    model_name = get_best_model()

    if model_name == 'XGBoost':
        forecast = handle_xgb_prediction(state, weeks)
    elif model_name == 'Prophet':
        forecast = predict_prophet(state, l_date, weeks)
    else:
        forecast = predict_arima(state, weeks)
    
    return forecast









































# test_main()




# def prepare_xgboost_data(df, predictions):
#     temp_df = df.copy()
#     temp_df.loc[len(temp_df)] = np.nan

#     pass












    # df['rolling_mean_1'] = df['Total'].transform(lambda x: x.shift(1).rolling(1).mean())
    # df['rolling_mean_7'] = df['Total'].transform(lambda x: x.shift(1).rolling(7).mean())
    # df['rolling_mean_30'] = df['Total'].transform(lambda x: x.shift(1).rolling(30).mean())
    # df['rolling_std_7'] = df['Total'].transform(lambda x: x.shift(1).rolling(7).std())
    # df['rolling_std_30'] = df['Total'].transform(lambda x: x.shift(1).rolling(30).std())