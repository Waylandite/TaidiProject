import pandas as pd
import matplotlib.pyplot as plt

def classify_by_season(data):
    data['order_date'] = pd.to_datetime(data['order_date'])
    season_map = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
    data['season'] = data['order_date'].dt.month.map(lambda x: season_map[(x % 12 + 3) // 3])
    demand_by_time = data.groupby('season')['ord_qty'].sum()

    # 绘制柱状图
    demand_by_time.plot(kind='bar')
    plt.show()

def classify_by_every_year_season(data):
    data['order_date'] = pd.to_datetime(data['order_date'])
    data['year'] = data['order_date'].dt.year
    for year in range(2015, 2019):
        data_year = data[data['year'] == year]
        season_map = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
        data_year['season'] = data_year['order_date'].dt.month.map(lambda x: season_map[(x % 12 + 3) // 3])
        demand_by_time = data_year.groupby('season')['ord_qty'].sum()

        # 绘制柱状图
        plt.title(str(year))
        demand_by_time.plot(kind='bar')
        plt.show()

if __name__ == '__main__':
    # 读取csv文件
    data = pd.read_csv('../数据/官方数据_完整版/order_train1.csv')
    # classify_by_season(data)
    classify_by_every_year_season(data)