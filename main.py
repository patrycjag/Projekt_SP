# Import required libraries
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Import necessary modules
from sklearn.model_selection import train_test_split

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true))


df = pd.read_csv('data/final.csv')
print(df.shape)
df.describe().transpose()

target_column = ['real']
predictors = ['l1', 'l2', 'l3',
              'l22', 'l23', 'l24', 'l25', 'l26',
              't3', 't5',
              'weekday',
              'yearday',
              'hour'
              ]
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
X = scaler.fit_transform(df[predictors].values)
y = scaler.fit_transform(df[target_column].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# 3.1 Single Multi-layer Neural Network
mlp = MLPRegressor(hidden_layer_sizes=(24, 8, 15), activation='relu', solver='adam',
                   max_iter=500, verbose=True, tol=0.0000001)
mlp.fit(X_train, y_train.ravel())

predict_test = mlp.predict(X_test)

print("3.1 Single Multi-layer Neural Network")
print("MAE = " + str(mean_absolute_error(y_test.ravel(), predict_test) * 100) + "%")
print("MAPE = " + str(mape(y_test.ravel(), predict_test) * 100) + "%")

print("3.2 Modular Neural System")
for h in range(1, 25):
    dfh = df[df['hour'] == h]
    target_column = ['real']
    predictors = ['l1', 'l2', 'l3',
                  'l22', 'l23', 'l24', 'l25', 'l26',
                  't3', 't5',
                  'weekday',
                  'yearday',
                  ]
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    X = scaler.fit_transform(dfh[predictors].values)
    y = scaler.fit_transform(dfh[target_column].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

    mlp = MLPRegressor(hidden_layer_sizes=(15, 8), activation='relu', solver='adam',
                       max_iter=5000, tol=0.0000001)
    mlp.fit(X_train, y_train.ravel())

    predict_test = mlp.predict(X_test)

    print("Hour " + str(h) + ":")
    print("MAE = " + str(mean_absolute_error(y_test.ravel(), predict_test) * 100) + "%")
    print("MAPE = " + str(mape(y_test.ravel(), predict_test) * 100) + "%")
    print("")

print("3.3 Committee Neural System")
K = 5
for h in range(1, 25):
    dfh = df[df['hour'] == h]
    target_column = ['real']
    predictors = ['l1', 'l2', 'l3',
                  'l22', 'l23', 'l24', 'l25', 'l26',
                  't3', 't5',
                  'weekday',
                  'yearday',
                  ]
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    X = scaler.fit_transform(dfh[predictors].values)
    y = scaler.fit_transform(dfh[target_column].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

    mlps = [MLPRegressor(hidden_layer_sizes=(15, 8), activation='relu', solver='adam',
                       max_iter=10000, tol=1E-7) for i in range(K)]

    print("Hour " + str(h) + ":")
    for i, mlp in enumerate(mlps):
        mlp.fit(X_train, y_train.ravel())
        ktest = mlp.predict(X_test)
        print("Learnt K #" + str(i) + ": MAPE = " + str(mape(y_test.ravel(), ktest) * 100) + "%")

    predict_tests = [mlp.predict(X_test) for mlp in mlps]

    predict_test = np.zeros_like(predict_tests[0])
    for pt in predict_tests:
        predict_test = predict_test + pt

    predict_test = predict_test / K

    print("MAE = " + str(mean_absolute_error(y_test.ravel(), predict_test) * 100) + "%")
    print("MAPE = " + str(mape(y_test.ravel(), predict_test) * 100) + "%")
    print("")
