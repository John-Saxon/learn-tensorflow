import pandas as pd


data1 = pd.DataFrame({
    'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'b': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
}).shift()

print(data1)


data2 = data1.dropna()
print(data2)

data3 = data1.dropna(axis=1)
print(data3)


data4 = data1.dropna(axis=1, subset=[0])
print(data4)

print('----------5------------')
data5 = data1.dropna(inplace=True)
print(data5)
print('----------1------------')
print(data1)
