import pandas as pd

print(pd.Categorical([1, 2, 3, 1, 2, 3]))
print(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c']))
print(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'],
                     ordered=True, categories=['c', 'b', 'a']))
print(pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c']))
