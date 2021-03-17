from backtest import *
from yahoo import *
from worker import *
import concurrent.futures
import itertools
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd


def showCAGRandCalmar(ticker, tickers_pool, prices_pool):
    tickers = [ticker]
    report_df_train, tickers = showResult(tickers, tickers_pool, prices_pool)
    result = report_df_train.loc[['CAGR', 'Calmar Ratio']]
    result['EqualWeight'] = result['EqualWeight'].str.strip('%')
    result = result.rename(columns = {'EqualWeight': ticker})
    return result


def runKMeanClustering(tickers_pool, beginDate, endDate, clusterSize,saveFigure = False, prefix = None, i = None):
    if i is None:
        i = ''
    prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, beginDate, endDate, clean_tickers = False,

                                                       common_dates = True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_list = executor.map(showCAGRandCalmar, tickers_pool, itertools.repeat(tickers_pool), itertools.repeat(
                prices_pool))
        result_list = list(result_list)
    print(f'\nCombine {len(result_list)} Calmar ratio and CAGR result')
    result = pd.concat(result_list, axis = 1)
    result = result.transpose()
    kmeans = KMeans(n_clusters = clusterSize, n_init = 30, max_iter = 3e8, tol = 1e-4, )
    kmeans.fit(result)
    labels = kmeans.predict(result)
    centroids = kmeans.cluster_centers_
    result['Class'] = labels
    result['Class'] = 'C' + result['Class'].astype(str)
    if saveFigure:
        fig = plt.figure()
        # plt.scatter(x = result['CAGR'].values.astype(float), y = result['Calmar Ratio'].values.astype(float))
        # plt.savefig('test')
        # plt.clf()
        plt.scatter(x = result['CAGR'].values.astype(float), y = result['Calmar Ratio'].values.astype(float),
                    color = result[
                        'Class'].values,
                    alpha = 0.3,
                    edgecolors = 'k')
        for idx, centroid in enumerate(centroids):
            plt.scatter(*centroid, color = f'C{idx}')
        # plt.xlim(-100, 100)
        # plt.ylim(-20, 20)
        plt.xlabel('CAGR')
        plt.ylabel('Calmar Ratio')
        if prefix is None:
            plt.savefig('kMeansClustering')
        else:
            if i is None:
                plt.savefig(f'./{prefix}/kMeansClustering')
            else:
                plt.savefig(f'./{prefix}/kMeansClustering_{i}')

    return result, centroids


if __name__ == '__main__':
    tickers_pool = createTickerpool(bool_ALL = False, bool_ETF = False, bool_SPX = True, extra_tickers = [])
    tickers_pool = removeUnwantedTickers_pool(tickers_pool, unwanted_tickers_list = None)
    beginDate = datetime.date(2015, 9, 3)
    endDate = datetime.date(2020, 9, 3)
    df_kMeanResult, centroids = runKMeanClustering(tickers_pool, beginDate, endDate, 5, showFigure = True)
