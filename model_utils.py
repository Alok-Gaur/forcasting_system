import math
import pandas as pd
import pickle
import json


def save_model(model, model_name, filename):
    filepath = f'models/{model_name}/{filename}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_name, filename):
    filepath = f'models/{model_name}/{filename}.pkl'
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        raise FileNotFoundError

def load_latest_data():
    data_versions_list = pd.read_json("Data_Version_History.json")
    latest_filename = data_versions_list['filename'].iloc[-1]
    filepath = f'version_data/{latest_filename}.csv'
    df = pd.read_csv(filepath)
    return df

def load_latest_encoding():
    data_versions_list = pd.read_json("Data_Version_History.json")
    latest_filename = data_versions_list['filename'].iloc[-1]
    filepath = f'encodings/{latest_filename}.pkl'
    with open(filepath, 'rb') as f:
        le = pickle.load(f)
    return le

def get_best_model():
    with open("Best_Model.json", "r") as f:
        model_details = json.load(f)
    return model_details['Model']




def split_data(df, split=0.2):
    train_list = []
    valid_list = []
    split_idx = math.ceil(df['State'].value_counts()[0]*split)
    for state in df['State'].unique():

        state_df = df[df['State'] == state]

        train_list.append(state_df.iloc[:-split_idx])
        valid_list.append(state_df.iloc[-split_idx:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    valid_df = pd.concat(valid_list).reset_index(drop=True)

    return train_df, valid_df

