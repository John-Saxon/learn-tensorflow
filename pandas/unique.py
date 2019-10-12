import pandas as pd
import numpy as np

batch = 1000

train_data_dir = '~/Downloads/new_ele_power_cur_vol_weather.load.Three.bias.csv'
reader = pd.read_csv(train_data_dir, chunksize=1000)


def MyUnique():
    df = next(reader)

    print(df.shape)
    # print(df.head(10))
    # print(df.dtypes)

    # u = pd.unique(df)

    print(type(df['STAT_DATE'].values))

    u1 = pd.unique(df['STAT_DATE'])

    # print(u)
    print(u1)
    print(type(u1))

    return u1


def NumpyArrayUnique():
    u1 = MyUnique()
    u2 = MyUnique()
    df = next(reader)
    # print(np.hstack((u1, u2)))
    u2 = pd.unique(np.hstack((u1, df['STAT_DATE'].values)))
    print(u2)
    # print(type(u2))


if __name__ == '__main__':
    # MyUnique()
    NumpyArrayUnique()
