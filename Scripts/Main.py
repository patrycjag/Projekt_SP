# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name='Rzeczywiste zapotrzebowanie KSE',
      column_names=['Data', 'Godz', 'Dobowa prognoza zapotrzebowania KSE', 'Rzeczywiste zapotrzebowanie KSE'],
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)

  return dataset


train_file_path = './../Data/JULY_2020.csv'
raw_train_data = get_dataset(train_file_path)

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


show_batch(raw_train_data)