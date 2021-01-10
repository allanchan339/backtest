import backtest
import datetime
tickers_pool = []
extra = ['GLD']
tickers_pool += extra
tickers_pool = list(set(tickers_pool))

beginDate = datetime.date(2015, 1, 1)
endDate = datetime.date(2020, 9, 15)

endDate = datetime.date.today()
prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, beginDate, endDate, clean_tickers = False, common_dates = True)
print(prices_pool)
# is_NaN = prices_pool.isnull()
# row_has_NaN = is_NaN.any(axis=1)
# rows_with_NaN = prices_pool[row_has_NaN]
# print(rows_with_NaN)
