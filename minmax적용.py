import tensorflow
import keras
import pandas as pd
import numpy as np


x_train = pd.read_csv("capp.csv")
y_train = pd.read_csv("capp_y.csv")
x_test = pd.read_csv("capp_test.csv")
y_test = pd.read_csv("capp_test_y.csv")

def minmax_norm(x_train_input):
    return (x_train - x_train.min()) / (x_train.max() - x_train.min())

x_train = minmax_norm(x_train)

def minmax_norm(y_train_input):
    return (y_train - y_train.min()) / (y_train.max() - y_train.min())

y_train = minmax_norm(y_train)

def minmax_norm(x_test_input):
    return (x_test - x_test.min()) / (x_test.max() - x_test.min())

x_test = minmax_norm(x_test)

def minmax_norm(y_test_input):
    return (y_test - y_test.min()) / (y_test.max() - y_test.min())

y_test = minmax_norm(y_test)

print(x_train)
print(y_train)
print(x_test)
print(y_test)