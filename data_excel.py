
import pandas as pd
import sklearn


def sklearn_csv_to_data(csv_file):
    df = pd.read_csv('capp.csv', header = 0)
    feature_names = list(df.columns.values)
    data = df.values
    return feature_names, data

feature_names, data = sklearn_csv_to_data('capp.csv')

print(data)

data['C1'] = data['C1'].astype(float)
print(data)

#csv 파일 행렬로 불러오기


#####################################################################
#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import xlrd

#book = pd.read_excel('C:/Users/tkdah/Desktop/deep learning/capp_excel.xlsx')
#print(book)

#def sklearn_excel_to_Data(excel_file):
#    df = pd.read_excel(input_file, header=0)
#    feature_names = list(df.colums.values)
#    data = df.valuers
#    return feature_names, data

#feature_names, data = sklearn_excel_to_Data("")

#############################################################excel 형식
#sheet = book.sheet_by_name('sheet1')
#data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
# Profit !
#print(data)


#x_train = pd.read_csv("capp.csv")
#y_train = pd.read_csv("capp_y.csv")
#x_test = pd.read_csv("capp_test.csv")
#y_test = pd.read_csv("capp_test_y.csv")