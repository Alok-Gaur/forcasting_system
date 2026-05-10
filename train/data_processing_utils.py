import pandas as pd
import holidays
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone
import pickle
# from processing_utils import split_data
import os

# Simple cleaning
def basic_clean(df):
    '''Take dataframe and removes null values,
    duplicates and converts date column to datetime'''
    data = df.copy()
    data = data.dropna()
    data = data.drop_duplicates()
    data.drop(columns=["Category","index", "Unnamed: 0"], inplace=True, errors='ignore')
    data['Total'] = data['Total'].astype(str).str.replace(',', '').str.strip().astype(float)
    data['Date'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True)
    data = data.sort_values(['State', 'Date'])
    return data

# Creating lag Attribute
def create_lag(df, lags=[1, 7, 30]):
    '''Take dataframe and lag values, create new lag columns

        Args:
            df: Dataframe = dataframe to be used
            lags: list|int = integer or list of integer of lags
    '''
    data = df.copy()

    if isinstance(lags, int):
        lag_col_name = f'lag_{lags}'
        data[lag_col_name] = data.groupby('State')['Total'].shift(lags)
    elif isinstance(lags, (list, tuple)):
        for lag in lags:
            lag_col_name = f'lag_{lag}'
            data[lag_col_name] = data.groupby('State')['Total'].shift(lag)
    else:
        raise ValueError('lags must be an integer or a list/tuple of integers')
    return data.sort_values(['State', 'Date'])


# Creating rolling mean and std
def rolling_mean_std(df, lags=[7, 30]):
    """ Take dataframe and lags value to create rolling mean and standard deviation """
    data = df.copy()
    if isinstance(lags, int):
        mean_col_name = f'rolling_mean_{lags}'
        std_col_name = f'rolling_std_{lags}'
        if lags != 1:
            data[mean_col_name] = data.groupby('State')['Total'].transform(lambda x: x.shift(1).rolling(lags).mean())
            data[std_col_name] = data.groupby('State')['Total'].transform(lambda x: x.shift(1).rolling(lags).std())
    elif isinstance(lags, (list, tuple)):
        for lag in lags:
            mean_col_name = f'rolling_mean_{lag}'
            std_col_name = f'rolling_std_{lag}'
            if lag != 1:
                data[mean_col_name] = data.groupby('State')['Total'].transform(lambda x: x.shift(1).rolling(lag).mean())
                data[std_col_name] = data.groupby('State')['Total'].transform(lambda x: x.shift(1).rolling(lag).std())
    else:
        raise ValueError('lags must be an integer or a list/tuple of integers')
    return data.sort_values(['State', 'Date'])

def date_feature_engineering(df):
    ''' Create feature related to date, eg. day of week, month and holiday flag'''
    data = df.copy()
    start_year, end_year = data['Date'].min().year, data['Date'].max().year
    vacations = holidays.US(years=range(start_year, end_year+1))
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.dayofweek
    data['holiday'] = data['Date'].apply(lambda x : 1 if x in vacations else 0)
    # data.dropna(inplace=True)
    return data

def encode_labels(df, col = 'State', filename=''):
    '''Encode the label in training and forcasting mode.
    "filename" is identifier of mode, if None-> Testing Mode
    '''
    # Loads encoder from version history and encode data
    if not filename:
        version_df = pd.read_json("Data_Version_History.json")
        encoder_filename = version_df['filename'].iloc[-1]
        with open(f'encodings/{encoder_filename}', 'rb') as f:
            le = pickle.load(f)
        df[col] = le.transform(df[col].astype(str))
    
    # Train encoder and store in encodings
    else:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        with open(f'encodings/{filename}.pkl', 'wb') as f:
            le = pickle.dump(le, f)
    return df
        
            


def training_data_processing(filepath, data_version=False):
    """ Preprocess the data and save it's version history (if set to True)

    Data Version History Fields:
        filename: stored data version file name.
        f_date: first date entry in the file
        l_date: laste date entry in the file
    """
    # identifying the correct file format and load
    file_type = filepath.split('.')[-1].lower()
    if file_type == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Applying Cleaning and feature engineering
    df = basic_clean(df)
    df = create_lag(df)
    df = rolling_mean_std(df)
    df = date_feature_engineering(df)

    # Drop null rows and unnecessary columns
    df.dropna(inplace=True)

    # Handle Data Version
    if data_version:
        filename = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
        df = encode_labels(df, col='State', filename=filename)
        # Save the csv file
        df.to_csv(f'version_data/{filename}.csv', index=False)

        # Save new file into data version record
        f_date = df['Date'].min().strftime('%Y-%m-%d')
        l_date = df['Date'].max().strftime('%Y-%m-%d')
        if os.path.isfile("Data_Version_History.json"):
            version_history = pd.read_json("Data_Version_History.json")
            temp_df = pd.DataFrame({'filename': [filename], 'f_date':[f_date] , 'l_date':[l_date]})
            version_history = pd.concat([version_history, temp_df], ignore_index=True)
            version_history.to_json("Data_Version_History.json")
        else:
            version_history = pd.DataFrame({'filename': [filename], 'f_date':[f_date] , 'l_date':[l_date]})
            version_history.to_json("Data_Version_History.json")
    else:
        df = encode_labels(df)
    return df
            

# training_data_processing(filepath=r"version_data\Forecasting Case- Study.xlsx - Sheet1.csv", data_version=True)




# def encode_labels(df, column=[], train=False):
#     le = LabelEncoder()
#     df[column] = le.fit_transform(df[column])
#     filename = str(datetime.now(timezone.utc).isoformat(timespec='seconds'))
    
#     try:
#         # Store label encodings
#         with open(f'encodings/{filename}.pkl', 'wb') as f:
#             pickle.dump(le, f)
        
#         df.to_csv(f"version_data/{filename}.csv")
#     except Exception as e:
#         print(f"Error while saving the encodings and data file:\n {e}")
    
#     return df


# def process_and_save_data(filename, encod_labels=True):
#     df = pd.read_csv(filename)
#     df = basic_clean(df)
#     df = create_lag(df)
#     df = rolling_mean_std(df)
#     df = date_feature_engineering(df)
#     obj_cols = [col for col in df.columns if df[col].dtype == 'object']
#     encode_labels(df, obj_cols)
#     if os.path.isfile("Data_Version_History.json"):
#         history_df = pd.read_json("Data_Version_History.json")
#         data_path_df = {"filepath": }


    