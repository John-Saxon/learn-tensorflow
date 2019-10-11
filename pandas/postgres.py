from sqlalchemy import create_engine
import pandas as pd

db_info = {
    'user': 'johnsaxon',
    'password': 'lavaadmin',
    'master': 'localhost1',
    'standby': 'localhost',
    'database': 'postgres',
    'port': 5432
}

engine = create_engine(
    "postgresql://%(user)s@%(master)s:%(port)d/%(database)s" % db_info, encoding='utf-8')

engine_bak = create_engine(
    "postgresql://%(user)s@%(standby)s:%(port)d/%(database)s" % db_info, encoding='utf-8')


def read_sql():
    hp = pd.read_sql_query(
        "select * from public.housing_predict", engine)

    print(hp.shape)
    print(hp.head())


def read_sql_with_chunk():
    iter_hp = pd.read_sql_query(
        "select * from public.housing_predict", engine, chunksize=1000)
    # hp = next(iter_hp)

    for hp in iter_hp:
        print(hp.shape)
        print(hp.head())


def read_table_with_chunk():
    # iter_hp = pd.read_sql_table(
    #     "housing_predict", engine, schema='public', chunksize=1000)
    iter_hp = pd.read_sql_table(
        "housing_predict", engine, chunksize=1000)

    for hp in iter_hp:
        print(hp.shape)
        print(hp.head())


if __name__ == '__main__':
    try:
        read_table_with_chunk()
    except Exception as e:
        print(e)
