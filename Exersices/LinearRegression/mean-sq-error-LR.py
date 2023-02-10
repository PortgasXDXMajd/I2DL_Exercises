import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = np.array(x_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)

x_test = np.array(x_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


n = 700
alpha = 0.0001

b = 0
m = 0

epochs = 0
while(epochs < 1000):
    y = b + m * x_train
    error = y - y_train
    mean_sq_er = np.nansum(error**2)
    mean_sq_er = mean_sq_er/n

    b = b - alpha * 2 * np.nansum(error)/n 
    m = m - alpha * 2 * np.nansum(error * x_train)/n
    epochs += 1
    if(epochs%10 == 0):
        print("Your MSE of ", epochs , " : " , mean_sq_er)


y_prediction = b + m * x_test
print('R2 Score:',r2_score(y_prediction,y_test))