import psycopg2
import pandas as pd


def _time_analyze_(func):
    from time import clock
    exec_times = 1

    def callf(*args, **kwargs):
        start = clock()
        for i in range(exec_times):
            r = func(*args, **kwargs)
        finish = clock()
        print("{:<20}{:10.6} s".format(func.__name__ + ":", finish - start))
        return r
    return callf


@_time_analyze_
def run(cur):
    # cur.execute("select * from elec where random()<0.00001;")
    cur.execute("select * from elec order by random() limit 4000;")
    # cur.execute("select * from elec order by random() ;")
    # rows = cur.fetchmany(2)
    rows = cur.fetchall()
    return rows


def batch_iter(cur, num=3):
    i = 0
    while i < num:
        i += 1
        cur.execute("select * from elec order by random() limit 4000;")
        yield pd.DataFrame(cur.fetchall())
    cur.close()


@_time_analyze_
def main():
    conn = psycopg2.connect(database="postgres", user="oushu",
                            password="lavaadmin", host="139.217.118.66", port="5432")
    # psql -h 139.217.118.66 -U oushu postgres
    # with conn.cursor() as cur:
    cur = conn.cursor()

    b_iter = batch_iter(cur, 5)

    for i in b_iter:
        print(i.head(2))

    conn.commit()
    conn.close()


def batch_iter(conn, num=3):
    cur = conn.cursor()
    i = 0
    while i < num:
        i += 1
        cur.execute("select * from elec order by random() limit 4000;")
        yield pd.DataFrame(cur.fetchall())
    cur.close()
    conn.commit()
    conn.close()


def test():
    conn = psycopg2.connect(database="postgres", user="oushu",
                            password="lavaadmin", host="139.217.118.66", port="5432")
    # psql -h 139.217.118.66 -U oushu postgres
    # with conn.cursor() as cur:

    b_iter = batch_iter(conn, 1)

    for i in b_iter:
        print(i.head(2))


if __name__ == "__main__":
    # main()
    test()
