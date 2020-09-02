from __future__ import absolute_import, division, print_function, unicode_literals
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# from six.moves import urllib
# import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

# amnesia = pd.read_csv('C:/Users/Dirani/Desktop/AI/homework/amnesiatest.csv' )
amnesia = pd.read_csv('amnesiatest.csv')
amnesiaeval = pd.read_csv('amnesiaeval.csv')
y_train = amnesia.pop('had')
y_eval = amnesiaeval.pop('has')
amnesia.pop('name')
#amnesia.pop('his_parents_have')
#amnesia.pop('photo_guy')
#amnesiaeval.pop('his_parents_have')
#amnesiaeval.pop('photo_guy')

print( amnesia )
print( amnesiaeval )
print( y_train )
print( y_eval )

CATEGORICAL_COLUMNS = [
  'sex'
        ]

NUMERIC_COLUMNS = [
  'age' , 'photo', 'his_parents_have'
                    ]

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = amnesia[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=5):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(amnesia, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(amnesiaeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train( train_input_fn )  # train
result = linear_est.evaluate( eval_input_fn )  # get model metrics/stats by testing on tetsing data
#print(result)
#clear_output()  # clears console output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model

predictionElement = linear_est.predict( eval_input_fn )
print( predictionElement )
result1 = list( predictionElement )
print("first predicted item")
print(result1[0])
print("second predicted item")
print(result1[1])
print("third predicted item")
print(result1[2])


print( amnesiaeval.loc[0] )
print( y_eval.loc[0] )
print( result1[0]['probabilities'][0] )
print( result1[0]['probabilities'][1] )
print( result1[0]['probabilities'][2] )

