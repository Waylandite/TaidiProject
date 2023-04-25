import  pandas as pd

import  matplotlib.pyplot as plt
import numpy as np
def paint_box():
    data=pd.read_csv("../数据/官方数据_完整版/order_train1.csv")
    # data.boxplot(column="ord_qty",by="item_code")
    plt.boxplot(data["ord_qty"])
    plt.title("ord_qty boxplot")
    plt.show()
    plt.boxplot(data["item_price"])
    plt.title("item_price boxplot")
    plt.show()

def three_sigma(Ser1):
    '''
    Ser1：表示传入DataFrame的某一列。
    '''
    rule = (Ser1.mean()-3*Ser1.std()>Ser1) | (Ser1.mean()+3*Ser1.std()< Ser1)
    index = np.arange(Ser1.shape[0])
    index=index[rule] #返回布尔值为True的索引
    return index  #返回落在3sigma之外的行索引值

def delete_out3sigma(data):
    out_index = [] #保存要删除的行索引
    # for i in range(data.shape[1]):  # 对每一列分别用3sigma原则处理
    index = three_sigma(data["ord_qty"])
    out_index += index.tolist()
    index = three_sigma(data["item_price"])
    out_index += index.tolist()
    delete_ = list(set(out_index))
    # print('所删除的行索引为：',delete_)
    print('删除前的数据量为：',len(delete_))
    data.drop(delete_,inplace=True)

    return data

def clear_data():
    data = pd.read_csv("../数据/官方数据_完整版/order_train1.csv")
    print(data.info())
    data=delete_out3sigma(data)
    print(data.info())
    data.to_csv("数据/order_train1_clear.csv",index=False)

def paint_clear_box():
    data=pd.read_csv("../数据/官方数据_清洗版/order_train1_clear.csv")
    # data.boxplot(column="ord_qty",by="item_code")
    plt.boxplot(data["ord_qty"])
    plt.title("ord_qty boxplot")
    plt.show()
    plt.boxplot(data["item_price"])
    plt.title("item_price boxplot")
    plt.show()

# 没有负数
def clear_negative_data():
    data = pd.read_csv("../数据/官方数据_清洗版/order_train1_clear.csv")
    print(data.info())
    data=data[data["ord_qty"]>0]
    data=data[data["item_price"]>0]
    print(data.info())
    data.to_csv("数据/order_train1_clear.csv",index=False)

if __name__=="__main__":
    # clear_data()
    # paint_clear_box()
    clear_negative_data()
    pass