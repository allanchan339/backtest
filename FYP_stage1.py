'''
This file is used to record result from different parameter and save result from GA result
'''
import numpy as np
from worker import *
def testModel(algorithm_param, tickers_size = 10, extra = []):
    # algorithm_param = create_algorithm_param(100,10,10,0.1,0.01,0.3,0.3)
    beginDate = datetime.date(2015, 9, 3)
    endDate = datetime.date(2020, 9, 3)
    # tickers_size = 10
    root_list_manual = None

    suffix = f'{algorithm_param["max_num_iteration"]}_{algorithm_param["population_size"]}_' \
             f'{algorithm_param["max_iteration_without_improv"]}_{algorithm_param["mutation_probability"]}_' \
             f'{algorithm_param["elit_ratio"]}_{algorithm_param["crossover_probability"]}_' \
             f'{algorithm_param["parents_portion"]}_E'
    createFolder(f'./{suffix}_{tickers_size}/')
    tickers, report_df_train, report_df_test, usedTime = runGA(tickers_size, beginDate, endDate, algorithm_param, suffix,
                                                               root_list_manual, extra)
    tickers_df = pd.DataFrame(tickers).transpose()
    usedTime = pd.DataFrame({"UsedTime": usedTime}.items())
    usedTime['UsedTime_str'] = usedTime[1].astype(str)
    usedTime = usedTime.drop(0, axis = 1).drop(1, axis = 1)
    print(tickers_df)
    print(report_df_train)
    print(usedTime)
    with pd.ExcelWriter(f'./{suffix}_{tickers_size}/stage1.xlsx', mode = 'A', datetime_format='hh:mm:ss.000') as writer:
        tickers_df.to_excel(writer, sheet_name='train', index=False)
        tickers_df.to_excel(writer, sheet_name='test', index=False)
        usedTime.to_excel(writer, sheet_name='train', index=False, startrow = 3)
        usedTime.to_excel(writer, sheet_name='test', index=False,startrow = 3)
        report_df_train.to_excel(writer, sheet_name = 'train', startrow = 7)
        report_df_test.to_excel(writer, sheet_name = 'test', startrow = 7)

def matrixCreation():
    result = []
    prefix = [0,1,2,3]
    substances = [0,0,0,0]
    for i in range(0,7):
        temp = []
        for j in range(0,7):
            if i == j:
                temp.append(prefix)
            else:
                temp.append(substances)
        result.append(temp)
        # print(np.array(result))
    return (np.array(result))
def matrixParser(matrix, index, subindex):
    matrix1 = matrix[index]
    matrix1 = np.transpose(matrix1)[subindex].tolist()
    print(matrix1)
    max_num_iterations = [100, 300, 500, 1000]
    population_sizes = [10, 30, 50, 100, 500]
    max_iteration_without_improvs = [10, 30, 50, 100]
    # multiprocessing_ncpuss = os.cpu_count()
    mutation_probabilitys = [0.1, 0.2, 0.3, 0.4]
    elit_ratios = [0.01, 0.02, 0.03, 0.04]
    crossover_probabilitys = [0.1,0.3,0.5,0.8]
    parents_portions = [0.1,0.3,0.5,0.8]
    algorithm_param = create_algorithm_param(max_num_iterations[matrix1[0]],population_sizes[matrix1[1]],
                                             max_iteration_without_improvs[matrix1[2]],mutation_probabilitys[matrix1[
                3]],elit_ratios[matrix1[4]],crossover_probabilitys[matrix1[5]],parents_portions[matrix1[6]])
    return algorithm_param

if __name__ == '__main__':
    matrix = matrixCreation()
    # for i in range(1,4):
    #     algorithm_param = matrixParser(matrix, 0, 0)
    algorithm_param = create_algorithm_param(100, 500, 10, 0.1, 0.01, 0.1, 0.1)
    # list = [8,10,12,14]
    # for i in list:
    extra = ['TMF', 'SOXX', 'MSFT', 'GLD', 'GBTC']
    testModel(algorithm_param, tickers_size = 10, extra = extra)

