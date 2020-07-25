import sqlalchemy
import bt
import pandas as pd
import numpy as np
import numba
import datetime
import concurrent.futures


def conn():
    username = 'root'  # 資料庫帳號
    password = 'root'  # 資料庫密碼
    host = 'localhost'  # 資料庫位址 #for safety reason i cant unlock to public network
    host = '192.168.1.50'
    # host = 'mysql'
    port = '3306'  # 資料庫埠號
    database = 'futu'  # 資料庫名稱
    # 建立連線引擎
    engine = sqlalchemy.create_engine(
            f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
            , echo = False, pool_size = 0, max_overflow = -1, pool_pre_ping = True, pool_recycle = 360)
    # ensure pool is pre_ping to check connection
    # no longer overflow for large insert
    # recycle session for 3600s to ensure session can survive to the end
    return engine


def readMysql(ticker, beginDate, endDate):
    sql = f''' SELECT Date, Close FROM `{ticker}` WHERE DATE(Date) >= '{beginDate}' AND DATE (Date) <= '{endDate}'
    '''
    engine = conn()
    df = pd.read_sql(sql, engine, index_col = 'Date')
    return df


def datafeedMysql(tickers, clean_tickers = True, common_dates = True):
    beginDate = datetime.date(2010, 1, 1)
    endDate = datetime.date(2020, 1, 1)
    endDate = datetime.date.today()
    index = pd.date_range(beginDate, endDate, freq = 'D')
    temp = pd.DataFrame(index = index)
    for ticker in tickers:
        df = readMysql(ticker, beginDate, endDate)
        if clean_tickers:
            df.rename(columns = {'Close': f'{ticker.lower()}'}, inplace = True)
        else:
            df.rename(columns = {'Close': f'{ticker}'}, inplace = True)
        temp = temp.merge(df, left_index = True, right_index = True)
    # with concurrent.futures.ProcessPoolExecutor as Executor:
    #     result = Executor.map(readMysql, tickers)
    if common_dates:
        temp = temp.dropna()
    return temp

def readReportCSV(report):
    temp = report.to_csv()
    temp = temp.replace(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,', '')
    temp = temp.splitlines()
    temp = list(filter(None, temp))
    csv = '\n'.join(temp)
    df = pd.DataFrame([x.split(',') for x in csv.split('\n')], columns = temp[0].split(','))
    df.drop(0, inplace = True)
    df.dropna(inplace = True)
    df.set_index(df['Stat'], inplace = True)
    df.drop(columns = 'Stat', inplace = True)
    df.index.name = None
    return df

def main():
    tickers = ['SPY', 'AGG']
    data = bt.get(tickers, start = '2010-01-01', )
    print(data)
    data0 = datafeedMysql(tickers)
    print(data0)


if __name__ == '__main__':
    main()
