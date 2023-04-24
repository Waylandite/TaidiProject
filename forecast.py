import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import datetime
from lightgbm import LGBMRegressor
from chinese_calendar import is_workday
import joblib
warnings.filterwarnings("ignore")
#数据读取
df = pd.read_csv('数据/order_train2.csv',encoding='utf-8')

#对价格进行分箱的函数
from scipy.stats import stats
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

#数据处理函数
def data_processing():
    #构造D列表，D列表是对应的天数
    dt1 = pd.date_range(start="20150901", end="20181220", freq="D")
    dicts = {}
    i = 1
    for date in dt1:
        dicts[str(date).split(' ')[0]] = i
        i += 1
    df['D'] = df['order_date'].map(dicts)
    #构造是否为促销的列表
    df['order_date'] = pd.to_datetime(df['order_date'])
    list_sales_promotion = ['1-1', '4-1', '2-14', '3-1', '3-8', '4-11', '5-1', '6-1', '6-18', '11-11', '12-12']
    sales_promotion = []
    for index,row in df.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in list_sales_promotion:
            sales_promotion.append(1)
        else:
            sales_promotion.append(0)
    df['promotion'] = sales_promotion
    #对价格进行分箱
    cut_bins = optimal_bins(df.ord_qty, df.item_price, n=10)
    df['price_range'] = pd.cut(df['item_price'], cut_bins, labels=[x for x in range(len(cut_bins) - 1)])
    # 销售渠道进行编码处理
    dict_qudao = {'offline': 0, 'online': 1}
    df.sales_chan_name = df.sales_chan_name.map(dict_qudao)
    # 对月初，月中、月末进行编码处理
    dict_moth = {'月初': 0, '月中': 1, '月末': 2}
    month = []
    for index, row in df.iterrows():
        if row['order_date'].day <= 10:
            month.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            month.append('1')
        elif row['order_date'].day > 20 :
            month.append('2')
    df['month_time_period'] = sales_promotion
    # 对季节进行编码处理
    dict_season = {'春': 0, '夏': 1, '秋': 2, '冬': 3}
    season=[]
    for index, row in df.iterrows():
        if row['order_date'].month==3 or row['order_date'].month==4 or row['order_date'].month==5:
            season.append('0')
        elif row['order_date'].month==6 or row['order_date'].month==7 or row['order_date'].month==8:
            season.append('1')
        elif row['order_date'].month==9 or row['order_date'].month==10 or row['order_date'].month==11:
            season.append('2')
        elif row['order_date'].month==12 or row['order_date'].month==1 or row['order_date'].month==2:
            season.append('3')
    df['seaon'] = season
    # 对星期几进行编码处理
    df['day_of_week'] = df.order_date.dt.weekday  # 0-6  {0, 1, 2, 3, 4, 5, 6}
    # 对是否工作日进行编码处理
    df['is_workday'] = df['order_date'].map(lambda x: is_workday(x))
    dict_is_workday = {False: 0, True: 1}
    df['is_workday'] = df.is_workday.map(dict_is_workday)
    #保存数据到tran2文件
    df.to_csv('数据/order_train2.csv', index=False, encoding='utf-8')

#构造2019年1月的数据
def Add201901():
    df['combination'] = df['sales_region_code'].astype(str) + '_' + df['first_cate_code'].astype(str) + '_' + df['second_cate_code'].astype(str) + '_' + df[
        'item_code'].astype(str)
    dt1 = pd.date_range(start="20181221", end="20190131", freq="D")
    dates = []
    need = []
    ids = []
    for d in dt1:
        date = str(d).split(' ')[0]  # 得到日期xx-xx-xx
        dates = dates + [date] * len(set(df['combination']))
        need = need + [0] * len(set(df['combination']))
        ids = ids + list(set(df['combination']))
    tempdf = pd.DataFrame()
    tempdf['order_date'] = dates
    tempdf['combination'] = ids
    tempdf['ord_qty'] = need
    tempdf['sales_region_code'] = tempdf['combination'].str.split('_', expand=True)[0]
    tempdf['first_cate_code'] = tempdf['combination'].str.split('_', expand=True)[1]
    tempdf['second_cate_code'] = tempdf['combination'].str.split('_', expand=True)[2]
    tempdf['item_code'] = tempdf['combination'].str.split('_', expand=True)[3]
    import chinese_calendar
    import datetime
    tempdf['order_date'] = pd.to_datetime(tempdf['order_date'])  # 对日期列进行日期格式转换
    # 构造是否为促销的列表
    tempdf['order_date'] = pd.to_datetime(tempdf['order_date'])
    list_sales_promotion = ['1-1', '4-1', '2-14', '3-1', '3-8', '4-11', '5-1', '6-1', '6-18', '11-11', '12-12']
    sales_promotion = []
    for index, row in tempdf.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in list_sales_promotion:
            sales_promotion.append(1)
        else:
            sales_promotion.append(0)
    tempdf['promotion'] = sales_promotion
    # 对价格进行分箱
    pricedf = df[['combination', 'price_range', 'sales_chan_name']]
    pricedf.drop_duplicates(['combination'], keep='last', inplace=True)
    dict_jg = {}
    qudao_dict = {}
    for i in range(len(pricedf)):
        dict_jg[list(pricedf['combination'])[i]] = list(pricedf['price_range'])[i]
        qudao_dict[list(pricedf['combination'])[i]] = list(pricedf['sales_chan_name'])[i]
    # 价格区间：
    jiage = []
    qudao = []
    for v in tempdf['combination']:
        jiage.append(dict_jg[v])
        qudao.append(qudao_dict[v])
    tempdf['price_range'] = jiage
    tempdf['sales_chan_name'] = qudao
    # 对月初，月中、月末进行编码处理
    dict_moth = {'月初': 0, '月中': 1, '月末': 2}
    month = []
    for index, row in tempdf.iterrows():
        if row['order_date'].day <= 10:
            month.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            month.append('1')
        elif row['order_date'].day > 20:
            month.append('2')
    tempdf['month_time_period'] = sales_promotion
    # 对季节进行编码处理
    dict_season = {'春': 0, '夏': 1, '秋': 2, '冬': 3}
    season = []
    for index, row in tempdf.iterrows():
        if row['order_date'].month == 3 or row['order_date'].month == 4 or row['order_date'].month == 5:
            season.append('0')
        elif row['order_date'].month == 6 or row['order_date'].month == 7 or row['order_date'].month == 8:
            season.append('1')
        elif row['order_date'].month == 9 or row['order_date'].month == 10 or row['order_date'].month == 11:
            season.append('2')
        elif row['order_date'].month == 12 or row['order_date'].month == 1 or row['order_date'].month == 2:
            season.append('3')
    tempdf['seaon'] = season
    # 对星期几进行编码处理
    tempdf['day_of_week'] = tempdf.order_date.dt.weekday  # 0-6  {0, 1, 2, 3, 4, 5, 6}
    # 对是否工作日进行编码处理
    tempdf['is_workday'] = tempdf['order_date'].map(lambda x: is_workday(x))
    dict_is_workday = {False: 0, True: 1}
    tempdf['is_workday'] =tempdf.is_workday.map(dict_is_workday)
    #添加D列表示天数
    dt1 = pd.date_range(start="20150902", end="20190131", freq="D")
    dicts = {}
    i = 1
    for date in dt1:
        dicts[str(date).split(' ')[0]] = i
        i += 1
    Ds = []
    for dat in tempdf['order_date']:
        Ds.append(dicts[str(dat).split(' ')[0]])
    tempdf['D'] = Ds
    # 保存数据到tran3文件
    tempdf.to_csv('数据/order_train3.csv', index=False, encoding='utf-8')
def feature_engineering():
    #滞后特征
    lags = [1, 2, 3, 6, 12, 24, 36, 48, 60]
    for lag in lags:
        df['need_lag_' + str(lag)] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', ], as_index=False)[
            'ord_qty'].shift(lag).astype(np.float16)
    #均值编码
    s = df.groupby('item_code')['ord_qty'].transform('mean')
    df['item_avg'] = df.groupby('item_code')['ord_qty'].transform('mean').astype(np.float16)
    df['sales_region_code_need_avg'] = df.groupby('sales_region_code')['ord_qty'].transform('mean').astype(np.float16)
    df['first_cate_code_need_avg'] = df.groupby('first_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    df['second_cate_code_need_avg'] = df.groupby('second_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    df['sales_chan_name_need_avg'] = df.groupby('sales_chan_name')['ord_qty'].transform('mean').astype(np.float16)
    df['sales_region_code_item_code_need_avg'] = df.groupby(['item_code', 'sales_region_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['first_cate_code_item_code_need_avg'] = df.groupby(['first_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['second_cate_code_item_code_need_avg'] = df.groupby(['second_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['sales_chan_name_item_code_need_avg'] = df.groupby(['sales_chan_name', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['sales_region_code_first_cate_code_need_avg'] = df.groupby(['sales_region_code', 'first_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['first_cate_code_second_cate_code_need_avg'] = df.groupby(['first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['sales_chan_name_second_cate_code_need_avg'] = df.groupby(['sales_chan_name', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    #滑动窗口统计
    df['rolling_need_mean'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.rolling(window=7).mean()).astype(np.float16)
    #开窗数据统计
    df['expanding_need_mean'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.expanding(2).mean()).astype(np.float16)
    #需求量趋势构建
    df['daily_avg_need'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(
        np.float16)
    df['avg_need'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['need_trend'] = (df['daily_avg_need'] - df['avg_need']).astype(np.float16)
    df.drop(['daily_avg_need', 'avg_need'], axis=1, inplace=True)
    df.drop('order_date', axis=1, inplace=True)
    #保存数据
    newdf = df[df['D']>=60]
    df.to_pickle('数据/train_data.pkl')

#构建模型
def model():
    data = pd.read_pickle('数据/train_data.pkl')
    valid = data[(data['D'] >= 1176) & (data['D'] < 1207)][['item_code', 'D', 'ord_qty']]
    test = data[data['D'] >= 1207][['item_code', 'D', 'ord_qty']]  # 作为待预测的数据
    eval_preds = test['ord_qty']  ## 待预测 目前全为0
    valid_preds = valid['ord_qty']  # 这是已有真实标签需求量1175到1206，31天间隔的真实数据
    states = [101, 102, 103, 104, 105]
    # 对五个销售区域分别建模并进行训练
    for store in states:
       try:
           df = data[data['sales_region_code'] == store]

           # Split the data
           X_train, y_train = df[df['D'] < 1176].drop('ord_qty', axis=1), df[df['D'] < 1176]['ord_qty']
           X_valid, y_valid = df[(df['D'] >= 1176) & (df['D'] < 1207)].drop('ord_qty', axis=1), \
                              df[(df['D'] >= 1176) & (df['D'] < 1207)]['ord_qty']
           X_test = df[df['D'] >= 1207].drop('ord_qty', axis=1)

           # Train and validate
           model = LGBMRegressor(
               n_estimators=1000,
               learning_rate=0.3,
               subsample=0.8,
               colsample_bytree=0.8,
               max_depth=8,
               num_leaves=50,
               min_child_weight=300
           )
           print('*****Prediction for 销售区域: {}*****'.format(store))
           model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                     eval_metric='rmse', verbose=20, early_stopping_rounds=20)
           valid_preds[X_valid.index] = model.predict(X_valid)
           eval_preds[X_test.index] = model.predict(X_test)
           filename = "模型/" + "model_" + str(store) + ".pkl"
           # save model
           joblib.dump(model, filename)
           del model, X_train, y_train, X_valid, y_valid
       except:
        del model, X_train, y_train, X_valid, y_valid
        continue
def test():
    model = joblib.load("模型/model_101.pkl")
    newdf = df[df['sales_region_code'] == 101]
    X_valid, y_valid = newdf[(df['D'] >= 1176) & (newdf['D'] < 1207)].drop('ord_qty', axis=1), \
                       newdf[(df['D'] >= 1176) & (newdf['D'] < 1207)]['ord_qty']
    s = model.predict(X_valid)
    x_axis = np.linspace(1, len(y_valid), len(y_valid))
    font1 = {'family': 'SimSun', 'weight': 'normal', 'size': 20, }
    colors = ['black', 'green', 'yellow', 'blue', 'cyan', 'red', 'orange', 'pink']
    plt.figure(dpi=300, figsize=(28, 12))

    plt.plot(x_axis[:200], y_valid[:200])
    plt.plot(x_axis[:200], s[:200])
    plt.legend(['true', 'prediction'], prop={'size': 20})
    # plt.scatter(x_data,y_data_yuemo,color='blue')
    plt.tick_params(labelsize=20)  # 修改刻度显示大小







if __name__ == '__main__':
    #model()
    test()
    pass

