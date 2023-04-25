import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 确定促销期
promotions = ['2015/6/18', '2015/11/11', '2016/6/18', '2016/11/11', '2017/6/18', '2017/11/11', '2018/6/18','2018/11/11']

def total_order():
    # 2. 加载并预处理数据
    df = pd.read_csv('../数据/官方数据_清洗版/order_train1_clear.csv', parse_dates=['order_date'], dtype={'sales_region_code': 'str'})
    df['is_promotion'] = df['order_date'].isin(promotions).astype(int)
    df_agg = df.groupby(['order_date'])['ord_qty'].sum().reset_index()

    # 3. 计算促销期和非促销期的订单需求量
    df_promo = df_agg[df_agg['order_date'].isin(promotions)]
    df_nonpromo = df_agg[~df_agg['order_date'].isin(promotions)]
    promo_mean = df_promo['ord_qty'].mean()
    nonpromo_mean = df_nonpromo['ord_qty'].mean()

    # 4. 可视化比较促销期和非促销期的订单需求量
    fig, ax = plt.subplots()
    ax.bar(['Promotion', 'Non-promotion'], [promo_mean, nonpromo_mean])
    ax.set_xlabel('Period')
    ax.set_ylabel('Average order quantity')
    ax.set_title('Effect of promotions on order quantity')
    plt.show()

def online_order():
    df = pd.read_csv('../数据/官方数据_清洗版/order_train1_clear.csv', parse_dates=['order_date'], dtype={'sales_region_code': 'str'})
    df=df[df["sales_chan_name"]=="online"]
    df['is_promotion'] = df['order_date'].isin(promotions).astype(int)
    df_agg = df.groupby(['order_date'])['ord_qty'].sum().reset_index()

    # 3. 计算促销期和非促销期的订单需求量
    df_promo = df_agg[df_agg['order_date'].isin(promotions)]
    df_nonpromo = df_agg[~df_agg['order_date'].isin(promotions)]
    promo_mean = df_promo['ord_qty'].mean()
    nonpromo_mean = df_nonpromo['ord_qty'].mean()

    # 4. 可视化比较促销期和非促销期的订单需求量
    fig, ax = plt.subplots()
    ax.bar(['Promotion', 'Non-promotion'], [promo_mean, nonpromo_mean])
    ax.set_xlabel('Period')
    ax.set_ylabel('Average order quantity')
    ax.set_title('Effect of promotions on online order quantity')
    plt.show()
def offline_order():
    df = pd.read_csv('../数据/官方数据_清洗版/order_train1_clear.csv', parse_dates=['order_date'], dtype={'sales_region_code': 'str'})
    df=df[df["sales_chan_name"]=="offline"]
    df['is_promotion'] = df['order_date'].isin(promotions).astype(int)
    df_agg = df.groupby(['order_date'])['ord_qty'].sum().reset_index()

    # 3. 计算促销期和非促销期的订单需求量
    df_promo = df_agg[df_agg['order_date'].isin(promotions)]
    df_nonpromo = df_agg[~df_agg['order_date'].isin(promotions)]
    promo_mean = df_promo['ord_qty'].mean()
    nonpromo_mean = df_nonpromo['ord_qty'].mean()

    # 4. 可视化比较促销期和非促销期的订单需求量
    fig, ax = plt.subplots()
    ax.bar(['Promotion', 'Non-promotion'], [promo_mean, nonpromo_mean])
    ax.set_xlabel('Period')
    ax.set_ylabel('Average order quantity')
    ax.set_title('Effect of promotions on offline order quantity')
    plt.show()

if __name__ == '__main__':
    total_order()
    online_order()
    offline_order()

