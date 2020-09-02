
from __future__ import absolute_import, division, print_function, unicode_literals
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# from six.moves import urllib
# import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

CSV_COLUMN_NAMES = ['head_ache', 'hard_breathing', 'what`s_up', 'unconsiousness']
Deseases = ['Covid 19', 'Brain Cancer', 'asphixia', 'feaver', 'Lung Cancer', 'nothing']
# Lets define some constants to help us later on
train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')
train_y = train.pop('what`s_up')
test_y = test.pop('what`s_up')  # the species column is now gone
"""
Covid 19, Brain Cancer, asphixia, feaver, Lung Cancer, nothing
0               1           2         3         4           5  
"""


def input_fn( features, labels, training=True, batch_size=4):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append( tf.feature_column.numeric_column( key=key ) )
print(my_feature_columns)

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=6)
classifier.train(
        input_fn=lambda: input_fn(train, train_y, training=True),
        steps=5000)  # We include a lambda to avoid creating an inner function previously
# eval_result = classifier.evaluate(
#     input_fn=lambda: input_fn(test, test_y, training=False))
#
# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

print("The old prediction.")
predictions = classifier.predict(
        input_fn=lambda: input_fn( test, test_y ) )
print( predictions )
# for pred_dict in predictions:
#     print(pred_dict)

feature_names = [ 'head_ache', 'hard_breathing', 'unconsiousness' ]

def input_fn_predict( feature_names, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    dataset = tf.data.Dataset.from_tensor_slices((dict(feature_names)))
    someobject = dataset.batch(batch_size)
    return someobject

import DeseasePredict
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)
DeseasePredict.predictdesease(feature_names, classifier, input_fn_predict, Deseases)

"""
expected = [0, 1, 2, 3, 4, 5]
predict_x = {'head_ache': [0, 1],
             'unconsiousness': [0, 1],
             'hard_breathing': [0, 1]}
"""