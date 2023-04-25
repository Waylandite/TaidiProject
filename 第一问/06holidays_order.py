import random

import numpy as np
import  pandas as pd
import numpy
import  matplotlib.pyplot as plt
import  matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime
import holidays
#通过折线图对比促销期间和非促销期间的平均需求量
def analayze_day_average():
    # 加载数据集并进行数据预处理
    data = pd.read_csv('../数据/官方数据_完整版/order_train1.csv')
    data['order_date'] = pd.to_datetime(data['order_date'])
    data['is_holiday'] = data['order_date'].isin(holidays.China(years=[2015, 2016, 2017, 2018]))
    data['is_holiday'] = data['is_holiday'].astype(int)
    # 计算促销和非促销期间的每天平均需求量
    holiday_mean_qty = data[data['is_holiday'] == 1].groupby('order_date')['ord_qty'].mean()
    nonholiday_mean_qty =data[data['is_holiday'] == 0].groupby('order_date')['ord_qty'].mean()
    # 可视化结果
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays',color='red')
    #ax.plot(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays', color='red')
    ax.plot(nonholiday_mean_qty.index, nonholiday_mean_qty.values, label='Non-Holidays')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Demand')
    ax.set_title('Impact of Holidays on Product Demand')
    ax.legend()
    plt.show()
def analayze_allyear_average():
        # 加载数据集并进行数据预处理
        data = pd.read_csv('../数据/官方数据_完整版/order_train1.csv')
        # 计算2016、2017、2018、2019三年的平均ord_qty数据
        data['order_date'] = pd.to_datetime(data['order_date'])
        a = []
        average = data['ord_qty'].mean()
        for i in range(597694):
            a.append(average)
        # 计算每个节假日的ord_qty数据
        data['is_holiday'] = data['order_date'].isin(holidays.China(years=[2015, 2016, 2017, 2018]))
        data['is_holiday'] = data['is_holiday'].astype(int)
        holiday_mean_qty = data[data['is_holiday'] == 1].groupby('order_date')['ord_qty'].mean()
        # 可视化结果
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays',color='red')
        #ax.plot(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays', color='red')
        # 将2016、2017、2018、2019三年的平均ord_qty数据画在图上
        ax.plot(data['order_date'], a, label='Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Demand')
        ax.set_title('Impact of Holidays on Product Demand')
        ax.legend()
        plt.show()
def analayze_2015year_average():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 加载数据集并进行数据预处理
    data = pd.read_csv('../数据/官方数据_完整版/order_train1.csv')
    data['order_date'] = pd.to_datetime(data['order_date'])
    #计算2016年的平均需求量
    data_2015 = data[data['order_date'].dt.year == 2015]
    average=data_2015['ord_qty'].mean()
    arr=[]
    for i in range(35726):
        arr.append(average)
    data_2015['is_holiday'] = data_2015['order_date'].isin(holidays.China(years=[2015]))
    data_2015['is_holiday'] = data_2015['is_holiday'].astype(int)
    holiday_mean_qty = data_2015[data_2015['is_holiday'] == 1].groupby('order_date')['ord_qty'].mean()
    #数据可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays', color='red')
    #在x轴上写出2016年国际节假日和中国节假日的名称
    for i in range(len(holiday_mean_qty.index)):
        if holiday_mean_qty.index[i].month == 1:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '元旦', fontsize=10)
        if holiday_mean_qty.index[i].month == 2:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '春节', fontsize=10)
        if holiday_mean_qty.index[i].month == 4:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '清明节', fontsize=10)
        if holiday_mean_qty.index[i].month == 5:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '劳动节', fontsize=10)
        if holiday_mean_qty.index[i].month == 6:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '端午节', fontsize=10)
        if holiday_mean_qty.index[i].month == 9:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '中秋节', fontsize=10)
        if holiday_mean_qty.index[i].month == 10:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '国庆节', fontsize=10)
        if holiday_mean_qty.index[i].month == 12:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '圣诞节', fontsize=10)
    ax.plot(data_2015['order_date'], arr, label='Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Demand')
    ax.set_title('Impact of Holidays on Product Demand in 2015')
    ax.legend()
    plt.show()
def analayze_2016year_average():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 加载数据集并进行数据预处理
    data = pd.read_csv('../数据/官方数据_完整版/order_train1.csv')
    data['order_date'] = pd.to_datetime(data['order_date'])
    #计算2016年的平均需求量
    data_2016 = data[data['order_date'].dt.year == 2016]
    average=data_2016['ord_qty'].mean()
    arr=[]
    for i in range(142894):
        arr.append(average)
    data_2016['is_holiday'] = data_2016['order_date'].isin(holidays.China(years=[2016]))
    data_2016['is_holiday'] = data_2016['is_holiday'].astype(int)
    data_holiday = data_2016[data_2016['is_holiday'] == 1]
    holiday_mean_qty = data_2016[data_2016['is_holiday'] == 1].groupby('order_date')['ord_qty'].mean()
    #数据可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays', color='red')
    #在x轴上写出2016年国际节假日和中国节假日的名称
    for i in range(len(holiday_mean_qty.index)):
        if holiday_mean_qty.index[i].month == 1:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '元旦', fontsize=10)
        if holiday_mean_qty.index[i].month == 2:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '春节', fontsize=10)
        if holiday_mean_qty.index[i].month == 4:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '清明节', fontsize=10)
        if holiday_mean_qty.index[i].month == 5:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '劳动节', fontsize=10)
        if holiday_mean_qty.index[i].month == 6:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '端午节', fontsize=10)
        if holiday_mean_qty.index[i].month == 9:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '中秋节', fontsize=10)
        if holiday_mean_qty.index[i].month == 10:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '国庆节', fontsize=10)
        if holiday_mean_qty.index[i].month == 12:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '圣诞节', fontsize=10)
    ax.plot(data_2016['order_date'], arr, label='Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Demand')
    ax.set_title('Impact of Holidays on Product Demand in 2016')
    ax.legend()
    plt.show()
def analayze_2017year_average():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 加载数据集并进行数据预处理
    data = pd.read_csv('../数据/官方数据_完整版/order_train1.csv')
    data['order_date'] = pd.to_datetime(data['order_date'])
    #计算2017年的平均需求量
    data_2017 = data[data['order_date'].dt.year == 2017]
    average=data_2017['ord_qty'].mean()
    arr=[]
    for i in range(196670):
        arr.append(average)
    data_2017['is_holiday'] = data_2017['order_date'].isin(holidays.China(years=[2017]))
    data_2017['is_holiday'] = data_2017['is_holiday'].astype(int)
    data_holiday = data_2017[data_2017['is_holiday'] == 1]
    holiday_mean_qty = data_2017[data_2017['is_holiday'] == 1].groupby('order_date')['ord_qty'].mean()
    #数据可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays', color='red')
    #在x轴上写出2017年国际节假日和中国节假日的名称
    for i in range(len(holiday_mean_qty.index)):
        if holiday_mean_qty.index[i].month == 1:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '元旦', fontsize=10)
        if holiday_mean_qty.index[i].month == 2:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '春节', fontsize=10)
        if holiday_mean_qty.index[i].month == 4:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '清明节', fontsize=10)
        if holiday_mean_qty.index[i].month == 5:
            if holiday_mean_qty.index[i].day == 1:
                ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '劳动节', fontsize=10)
            if holiday_mean_qty.index[i].day == 30:
                ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '端午节', fontsize=10)
        if holiday_mean_qty.index[i].month == 9:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '中秋节', fontsize=10)
        if holiday_mean_qty.index[i].month == 10:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '国庆节', fontsize=10)
        if holiday_mean_qty.index[i].month == 12:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '圣诞节', fontsize=10)
    ax.plot(data_2017['order_date'], arr, label='Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Demand')
    ax.set_title('Impact of Holidays on Product Demand in 2017')
    ax.legend()
    plt.show()
def analayze_2018year_average():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 加载数据集并进行数据预处理
    data = pd.read_csv('../数据/官方数据_完整版/order_train1.csv')
    data['order_date'] = pd.to_datetime(data['order_date'])
    #计算2018年的平均需求量
    data_2018 = data[data['order_date'].dt.year == 2018]
    average=data_2018['ord_qty'].mean()
    arr=[]
    for i in range(222404):
        arr.append(average)
    data_2018['is_holiday'] = data_2018['order_date'].isin(holidays.China(years=[2018]))
    data_2018['is_holiday'] = data_2018['is_holiday'].astype(int)
    data_holiday = data_2018[data_2018['is_holiday'] == 1]
    holiday_mean_qty = data_2018[data_2018['is_holiday'] == 1].groupby('order_date')['ord_qty'].mean()
    #数据可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(holiday_mean_qty.index, holiday_mean_qty.values, label='Holidays', color='red')
    #在x轴上写出2019年国际节假日和中国节假日的名称
    for i in range(len(holiday_mean_qty.index)):
        if holiday_mean_qty.index[i].month == 1:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '元旦', fontsize=10)
        if holiday_mean_qty.index[i].month == 2:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '春节', fontsize=10)
        if holiday_mean_qty.index[i].month == 4:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '清明节', fontsize=10)
        if holiday_mean_qty.index[i].month == 5:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '劳动节', fontsize=10)
        if holiday_mean_qty.index[i].month == 6:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '端午节', fontsize=10)
        if holiday_mean_qty.index[i].month == 9:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '中秋节', fontsize=10)
        if holiday_mean_qty.index[i].month == 10:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '国庆节', fontsize=10)
        if holiday_mean_qty.index[i].month == 12:
            ax.text(holiday_mean_qty.index[i], holiday_mean_qty.values[i], '圣诞节', fontsize=10)
    ax.plot(data_2018['order_date'], arr, label='Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Demand')
    ax.set_title('Impact of Holidays on Product Demand in 2018')
    ax.legend()
    plt.show()


if __name__=="__main__":
    #analayze_day_average()
    #analayze_allyear_average()
    analayze_2015year_average()
    analayze_2016year_average()
    analayze_2017year_average()
    analayze_2018year_average()
    pass