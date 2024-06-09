import pandas as pd
import numpy as np

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

if __name__ == "__main__":
    data = load_data('../data/sensor_data.csv')
    normalized_data = normalize_data(data)
    normalized_data.to_csv('../data/normalized_sensor_data.csv', index=False)

