from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from train.model_evaluation import evaluate_model
import pandas as pd
from train.data_processing_utils import *
from model_utils import split_data, save_model
import pickle


def train_arima(states, train_df, val_df=None):
    """This trains the train and validation data to forcast
        
        Args: 
            states:List[str] = List of all states
            train_df = Training dataframe
            val_df = validation dataframe
    """
    arima_preds = []
    if val_df is not None and not val_df.empty:
        for state in states:
            train_state = train_df[train_df['State'] == state]
            valid_state = val_df[val_df['State'] == state]

            train_series = train_state['Total'].values

            model = ARIMA(train_series, order=(1,1,1))
            model_fit = model.fit()

            forecast = model_fit.forecast(
                steps=len(valid_state)
            )

            arima_preds.extend(forecast)
            # Save the model
            save_model(model_fit, 'ARIMA', f"state_{str(state)}")

        arima_result = evaluate_model(
            val_df['Total'],
            arima_preds,
            "ARIMA"
        )

        arima_result['filepath'] ="models/ARIMA/"

        return arima_result
    else:
        for state in states:
            train_state = train_df[train_df['State'] == state]
            train_series = train_state['Total'].values

            model = ARIMA(train_series, order=(1,1,1))
            model_fit = model.fit()
            # Save the model
            save_model(model_fit, 'ARIMA', f"state_{str(state)}")
        
        return "Trained on whole data"



def train_prophet(states, train_df, val_df=None):
    prophet_preds = []
    if val_df is not None and not val_df.empty:
        for state in states:

            train_state = train_df[
                train_df['State'] == state
            ][['Date', 'Total']]

            valid_state = val_df[
                val_df['State'] == state
            ][['Date', 'Total']]

            prophet_train = train_state.rename(
                columns={
                    'Date': 'ds',
                    'Total': 'y'
                }
            )

            model = Prophet(daily_seasonality=False)
            # model.add_regressor('holiday')

            model.fit(prophet_train)

            future = valid_state.rename(
                columns={'Date': 'ds'}
            )

            forecast = model.predict(future)

            prophet_preds.extend(
                forecast['yhat'].values
            )

            save_model(model, 'Prophet', f"state_{str(state)}")

        prophet_result = evaluate_model(
            val_df['Total'],
            prophet_preds,
            "Prophet"
        )
        prophet_result['filepath'] = 'models/Prophet/'
        return prophet_result
    else:
        for state in states:
            train_state = train_df[
                train_df['State'] == state
            ][['Date', 'Total']]

            prophet_train = train_state.rename(columns={
                                                'Date': 'ds',
                                                'Total': 'y'
                                            }
                                        )

            model = Prophet(daily_seasonality=False)
            model.fit(prophet_train)

            save_model(model, 'Prophet', f"state_{str(state)}")
        
        return "Trained on whole data"




def train_xgboost(train_df, val_df=None):
    if val_df is not None and not val_df.empty:
        y_train = train_df['Total']
        X_train = train_df.drop(columns='Total')

        y_valid = val_df['Total']
        X_valid = val_df.drop(columns='Total')

        xgb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        xgb_model.fit(X_train, y_train)

        xgb_preds = xgb_model.predict(X_valid)

        xgb_result = evaluate_model(
            y_valid,
            xgb_preds,
            "XGBoost"
        )
        # filepath = f"models/XGBoost.pkl"
        # with open(filepath, 'wb') as f:
        #     pickle.dump(xgb_model, f)
        save_model(xgb_model, 'XGBoost', 'XGBoost')
        xgb_result['filepath'] = 'models/XGBoost/'
        return xgb_result
    else:
        y_train = train_df['Total']
        X_train = train_df.drop(columns='Total')

        xgb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        xgb_model.fit(X_train, y_train)
        save_model(xgb_model, 'XGBoost', 'XGBoost')
        return "Trained on whole data"
            
            


def train_on_entire_data(model_name='XGBoost', df=None):
    ''' Train the model on the entire dataset instead of splitting it into training and validation

        Args:
            model_name: str = options ['ARIMA', 'Prophet', 'XBGoost', 'all']
            df: optional[pd.DataFrame] = Data on which model will train on
    '''
    # load the latest dataframe if data not provided
    if not df:
        data_versions_list = pd.read_json("Data_Version_History.json")
        latest_filename = data_versions_list['filename'].iloc[-1]
        filepath = f'version_data/{latest_filename}.csv'

        df = pd.read_csv(filepath)

    df = basic_clean(df)
    if model_name == 'ARIMA' or model_name == 'all':
        train_arima(df['State'].unique(), df)
    if model_name == 'Prophet' or model_name == 'all':
        train_prophet(df['State'].unique(), df)
    if model_name == 'XGBoost' or model_name == 'all':
        temp_df = df.copy()
        temp_df.drop(columns=['Date'], inplace=True)
        train_xgboost(temp_df)
    if model_name in ('ARIMA', 'Prophet', 'XGBoost', 'all'):
        return True
    return False

# new_train(prophet=False, xgboost=False)



def evaluate_best_model(filename=None):
    '''Prepare appropiate data and pass it to the model'''
    # load latest version of data if filename is not provided
    if not filename:
        data_versions_list = pd.read_json("Data_Version_History.json")
        filename = data_versions_list['filename'].iloc[-1]

    filepath = f'version_data/{filename}.csv'
    df = pd.read_csv(filepath)
    df = basic_clean(df)

    results = []
    
    # 1. Train ARIMA Model
    # date_feature_engg_df = date_feature_engineering(basic_clean_df)
    train_df, val_df = split_data(df)
    arima_data = train_arima(df['State'].unique(), train_df, val_df)
    results.append(arima_data)
    
    # 2. Train Prophet Model
    # date_feature_engg_df = date_feature_engineering(basic_clean_df)
    train_df, val_df = split_data(df)

    prophet_data = train_prophet(df['State'].unique(), train_df, val_df)
    results.append(prophet_data)
    
    # 3. Train XGBoost Model
    df.drop(columns='Date', inplace=True)
    train_df, val_df = split_data(df)

    xgb_data = train_xgboost(train_df, val_df)
    results.append(xgb_data)


    # collect all the results
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        'RMSE'
    ).reset_index(drop=True)


    best_model = results_df.iloc[0]
    best_model_name = results_df.loc[0, 'Model']
    best_model.to_json("Best_Model.json")
    return results_df



# def test_code(filename="version_data/Forecasting Case- Study.xlsx - Sheet1.csv"):
#     results = []
#     df = pd.read_csv(filename)
#     results.append(train_model(df, 'arima'))
#     results.append(train_model(df, 'prophet'))
#     results.append(train_model(df, 'xgboost'))
#     results_df = pd.DataFrame(results)

#     results_df = results_df.sort_values(
#         'RMSE'
#     ).reset_index(drop=True)


#     print(results_df)

#     best_model = results_df.iloc[0]
#     best_model.to_json("Best_Model.json")











# def run_arima(filename="version_data/Forecasting Case- Study.xlsx - Sheet1.csv"):
#     df = pd.read_csv(filename)
#     basic_clean_df = basic_clean(df)
#     date_feature_engg_df = date_feature_engineering(basic_clean_df)
#     train_df, val_df = split_data(date_feature_engg_df)

#     data = train_arima(df['State'].unique(), train_df, val_df)
#     return data

# def run_prophet(filename="version_data/Forecasting Case- Study.xlsx - Sheet1.csv"):
#     df = pd.read_csv(filename)
#     basic_clean_df = basic_clean(df)
#     date_feature_engg_df = date_feature_engineering(basic_clean_df)
#     train_df, val_df = split_data(date_feature_engg_df)

#     data = train_prophet(df['State'].unique(), train_df, val_df)
#     return data

# def run_xgboost(filename="version_data/Forecasting Case- Study.xlsx - Sheet1.csv"):
#     df = pd.read_csv(filename)
#     basic_clean_df = basic_clean(df)
#     basic_clean_df = create_lag(basic_clean_df)
#     basic_clean_df = rolling_mean_std(basic_clean_df)
#     date_feature_engg_df = date_feature_engineering(basic_clean_df)
#     encoded_df = encode_labels(date_feature_engg_df)
#     encoded_df.drop(columns='Date', inplace=True)
#     train_df, val_df = split_data(encoded_df)

#     data = train_xgboost(train_df, val_df)
#     return data