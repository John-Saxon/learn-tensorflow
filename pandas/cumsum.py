import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4]],
                  columns=list('AB'))

print(df)
print()
print(df.cumsum())
print()
print(df.cumsum(axis=0))
print()
print(df.cumsum(axis=1))
print()
print(df.cumsum(axis=-1))
print()
print(df.cumsum(axis=2))
print()
print(df.cumsum(axis=-2))
