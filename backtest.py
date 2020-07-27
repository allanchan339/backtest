import sqlalchemy
import bt
import pandas as pd
import numpy as np
import numba
import datetime
import concurrent.futures
import pandas_market_calendars
import itertools

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


def createTradingpandas(beginDate, endDate):
    nyse = pandas_market_calendars.get_calendar('NYSE')
    schedule = nyse.schedule(start_date = beginDate, end_date = endDate)
    temp = pd.DataFrame(index = schedule.index)
    return temp


def readMysql(ticker, beginDate, endDate, clean_tickers = True):
    sql = f''' SELECT Date, Close FROM `{ticker}` WHERE DATE(Date) >= '{beginDate}' AND DATE (Date) <= '{endDate}'
    '''
    engine = conn()
    df = pd.read_sql(sql, engine, index_col = 'Date')
    engine.dispose()  # a good practice to kill engine
    if clean_tickers:
        df.rename(columns = {'Close': f'{ticker.lower()}'}, inplace = True)
    else:
        df.rename(columns = {'Close': f'{ticker}'}, inplace = True)
    return df


def datafeedMysql(tickers, beginDate, endDate, clean_tickers = True, common_dates = True):
    temp = createTradingpandas(beginDate, endDate)
    df_list = []  # df_dict = {}
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for i, file in zip(tickers, executor.map(readMysql, tickers, [beginDate] * len(tickers), [endDate] * len(
    #             tickers), [clean_tickers]*len(tickers))):
    #         df_list.append(file)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        df_list = executor.map(readMysql, tickers, itertools.repeat(beginDate), itertools.repeat(endDate),
                               itertools.repeat(clean_tickers))
        df_list = list(df_list) # is a bad written method ?

    for df in df_list:
        temp = temp.join(df)
    if common_dates:
        temp.dropna(inplace = True)
    return temp


def readReportCSV(report):
    temp = report.to_csv()
    temp = temp.replace(',,', '')  # drop if the line is blank only
    temp = temp.splitlines()
    temp = list(filter(None, temp))
    csv = '\n'.join(temp)
    df = pd.DataFrame([x.split(',') for x in csv.split('\n')], columns = temp[0].split(','))
    df.drop(0, inplace = True)
    df.dropna(inplace = True)
    df.set_index(df['Stat'], inplace = True)
    df.drop(columns = 'Stat', inplace = True)
    df.index.name = None
    df.dropna(inplace = True)
    return df


def replaceToYahooSymbolSyntax(word):
    return word.replace('.', '-')


def SPXlist():
    SPX_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    SPX_table = SPX_table[0]
    SPX_symbol = SPX_table['Symbol'].values.tolist()
    with concurrent.futures.ProcessPoolExecutor() as executor:  # max_workers = 24 on default
        SPX_symbol = executor.map(replaceToYahooSymbolSyntax, SPX_symbol)
    SPX_symbol = list(SPX_symbol)
    return SPX_symbol


def ETFlist():
    htmls = ['https://etfdb.com/compare/market-cap/', 'https://etfdb.com/compare/volume/']
    symbols = []
    for html in htmls:
        table = pd.read_html(html)
        df = table[0]
        symbol = df['Symbol'].values.tolist()
        symbols += symbol
    return symbols


def main():
    tickers = ['SPY', 'AGG']
    data = bt.get(tickers, start = '2010-01-01', )
    print(data)
    data0 = datafeedMysql(tickers)
    print(data0)


if __name__ == '__main__':
    main()
