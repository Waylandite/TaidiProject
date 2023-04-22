import pandas as pd
import matplotlib.pyplot as plt


def classify_by_month(data):
    data['order_date'] = pd.to_datetime(data['order_date']).dt.day
    data['order_date_category'] = pd.cut(data['order_date'], bins=[0, 10, 20, 31], labels=['begin','mid', 'end'])

    demand_by_time = data.groupby('order_date_category')['ord_qty'].sum()

    # 绘制柱状图
    demand_by_time.plot(kind='bar')
    plt.show()

def classify_by_year(data):
    data['order_date'] = pd.to_datetime(data['order_date']).dt.month

    demand_by_time = data.groupby('order_date')['ord_qty'].sum()

    # 绘制柱状图
    demand_by_time.plot(kind='bar')
    plt.show()

if __name__ == '__main__':
    # 读取csv文件
    data = pd.read_csv('数据/order_train1.csv')

    classify_by_month(data)
    classify_by_year(data)


