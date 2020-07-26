import bt
import backtest
import matplotlib
import pandas as pd
import platform
import matplotlib.pyplot as plt
import datetime


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

if platform.system() != 'Linux': #to show plt
    matplotlib.use('tkagg')

class GeniticAlgo: #nothing to inheritance
    #multi processing inside the class?
    def __init__(self, tickers_GA, prices_pool):
        self.tickers_GA = tickers_GA
        self.prices_pool = prices_pool

    def extractPrices(self, prices_pool):
        pass


startTime = datetime.datetime.now()
tickers = backtest.SPXlist()
tickers += backtest.ETFlist()
extra = ['TMF', 'SOXL', 'ARKW', 'ARKK', 'SMH', 'SOXX']
tickers += extra
tickers = list(set(tickers))

beginDate = datetime.date(2010, 1, 1)
endDate = datetime.date.today()

prices_pool = backtest.datafeedMysql(tickers, beginDate, endDate, clean_tickers = False, common_dates = False)
print(prices_pool)
finishTime = datetime.datetime.now()
print(f'We need {finishTime - startTime} to collect data')
tickers_GA = []
#start GA
#workflow
#random get ticker, flexible number, to 1000 father??
#GA to optim, step store into mysql algo, mutant gen with prob.
#use calmer ratio as my obj function
#5% from max - min , stop ?
#alert then show the top5 outcome, and display result?
