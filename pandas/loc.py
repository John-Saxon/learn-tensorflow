# https://www.cnblogs.com/hellcat/p/7476854.html
import pandas as pd

file = "../data/industry_timeseries/timeseries_predict_data/1.csv"
columns = ['YEAR', 'MONTH', 'DAY', 'TEMP_HIG',
           'TEMP_COL', 'AVG_TEMP', 'AVG_WET', 'DATA_COL']
df = pd.read_csv(file, names=columns)

print(df.shape)
print(df.head(5))


# loc must assign column names
print("\n\n-----------loc-------------\n")
print(df.loc[0, 'YEAR'])
print(df.loc[0:3, ['MONTH', 'DAY']])
print(df.loc[[0, 5],  ['MONTH', 'DAY']])


# iloc must assign column index
print("\n\n-----------iloc-------------\n")
print(df.iloc[1, 1])
print(df.iloc[0:3, [0, 1]])


# ix assign index or name
print("\n\n-----------ix-------------\n")
print(df.ix[1, 'YEAR'])
print(df.ix[0:3, [0, 1]])


print(df.ix[df.shape[0]-2: df.shape[0]+2])
