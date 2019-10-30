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


# @_time_analyze_
def run(cur):
    # cur.execute("select * from elec where random()<0.00001;")
    cur.execute("select * from elec order by random() limit 4000;")
    # cur.execute("select * from elec order by random() ;")
    # rows = cur.fetchmany(2)
    rows = cur.fetchall()

    df = pd.DataFrame(rows)
    return df


@_time_analyze_
def main():
    conn = psycopg2.connect(database="postgres", user="oushu",
                            password="lavaadmin", host="139.217.118.66", port="5432")
    # psql -h 139.217.118.66 -U oushu postgres
    cur = conn.cursor()

    for i in range(5):
        rows = run(cur)
        print(rows.head(2))

    conn.commit()
    conn.close()


if "__main__" == __name__:
    main()
