import pandas as pd


data1 = pd.DataFrame({
    'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'b': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
})

print("data1:\n", data1)
print(data1.shape)


data2 = data1.shift(axis=0)
print("data2:\n", data2)
print("data1 after shift:\n", data1)

data3 = data1.shift(axis=1)
print("data3:\n", data3)


data4 = data1.shift(periods=-1)
print("data4:\n", data4)


entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})

print((entries_and_exits-entries_and_exits.shift(axis=0)).fillna(0))
