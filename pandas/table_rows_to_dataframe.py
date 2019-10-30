import pandas as pd


rows = [
    ('a', 1, 7.0),
    ('b', 11, 7.5),
    ('c', 51, 9.0),
]

data = pd.DataFrame(rows, columns=["name",  "age", "other"], )

print(data)
print(data.dtypes)
