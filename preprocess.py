import pandas as pd
import numpy as np
import tensorflow as tf


def read_data(f_name):
    

    return pd.read_csv(f_name)



def OHE(df, col_name):
    values = df[col_name]
    unique = pd.unique(values)

    #unique = pd.unique(df[col_name])
    #print(unique)
    mapped = {}
    i = 0
    for iter in unique:

        mapped[iter] = i
        i = i + 1
    print(i)
    OHE_list = []

    for i in range(len(df)):
        
        OHE_list.append(mapped[df[col_name][i]])
    
    return OHE_list

        

'''
Converts coordinates in the form (x, y) into a singular value

Width is defaulted to 8 for purposes of a chess board
'''
def calculate_1d_position(x, y, width=8):
    return ((y-1) * 8) + (x - 1)

def normalise_x_y_coordinates(df, x_col_name, y_col_name):
    x_pos = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8
    }
    
    #print(df.keys())
    encoded_elements = []
    for i in range(len(df)):
        x = x_pos[df[x_col_name][i]]
        y = df[y_col_name][i]
        
        encoded_elements.append(calculate_1d_position(x, y))

    return encoded_elements

    
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('target')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

def distance(df, r1, r2):
    dist = []

    for i in range(len(df)):
        dist.append(abs(df[r1][i] - df[r2][i]))

    return dist