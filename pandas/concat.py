import pandas as pd

s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])
print(pd.concat([s1, s2]))
print(pd.concat([s1, s2], axis=1))
print(pd.concat([s1, s2], ignore_index=True))
