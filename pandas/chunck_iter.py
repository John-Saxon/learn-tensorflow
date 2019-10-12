import pandas as pd

batch = 1000

train_data_dir = '~/Downloads/new_ele_power_cur_vol_weather.load.Three.bias.csv'
reader = pd.read_csv(train_data_dir, chunksize=1000)

df = next(reader)

print(df.shape)
print(df.head(10))
print(df.dtypes)
