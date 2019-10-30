
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine(
    "postgresql://oushu:lavaadmin@139.217.118.66:5432/postgres", encoding='utf-8')
sql = "select * from elec where random<0.00001;"
iter1 = pd.read_sql_query(sql, engine)
data = next(iter1)
data.head()
