#对价格进行分箱的函数，返回分箱边界值列表
#因为与数据处理无关，所以单独移动到此文件中

from scipy.stats import stats
import pandas as pd
import numpy as np
def optimal_bins(Y, X, n):
    """
    :Y  目标变量
    ：X  待分箱特征
    ：n 分箱数初始值
    return : 统计值，分箱边界值列表、woe值、iv值
    """
    r = 0  # xia相关系数的初始值
    total_bad = Y.sum()  # 总的坏样本数
    total_good = Y.count() - total_bad  # 总的好样本数
    # 分箱过程
    while np.abs(r) < 1:  # 相关系数的绝对值等于1结束循环，循环目的找寻最好正反相关性
        # df1中的bin为给X分箱对应的结果
        df1 = pd.DataFrame({'X': X, 'Y': Y, 'bin': pd.qcut(X, n, duplicates='drop')})  # drop表示删除重复元素
        # 将df1基于箱子进行分组
        df2 = df1.groupby('bin')
        # r返回的是df1对箱子分组后，每组数据X的均值的相关系数，如果系数不为正负1，则减少分箱的箱数
        r, p = stats.spearmanr(df2.mean().X, df2.mean().Y)  # 计算相关系数
        n = n - 1
    cut = [0]  # 分箱边界值列表
    for i in range(1, n + 2):  # i的取值范围是1->（n+1），n+1是分箱的数量
        qua = X.quantile(i / (n + 1))  # quantile把给定的乱序的数值有小到大并列分成n等份，参数表述取出第百分之多少大小的数值
        # i的取值范围是1->n 1/n.2/n 3/n...n/n
        cut.append(round(qua, 6))
    return cut
