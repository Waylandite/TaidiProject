import random

import numpy as np
import  pandas as pd
import numpy
import  matplotlib.pyplot as plt
import  matplotlib



def cout_prices():
    data_train = pd.read_csv("数据/order_train1.csv")
    item_code_list = data_train.item_code.unique()
    print(len(item_code_list))
    map={}
    for row in data_train.itertuples():
        if row.item_code not in map.keys():
            map[row.item_code]={}
        if row.item_price not in map[row.item_code]:
            map[row.item_code][row.item_price]=row.ord_qty
        else:
            map[row.item_code][row.item_price]+=row.ord_qty
    for key in map.keys():
        map[key]=sorted(map[key].items(),key=lambda x:x[0])
    data=pd.DataFrame(map.items(),columns=['item_code','priceorder_list'])
    data.to_csv("数据/price_order.csv",index=False)
    # for key in map.keys():
    #     map[key]=len(map[key])
    # a = sorted(map.items(), key=lambda x: x[1])
    # print(a)
    # data = pd.DataFrame(map.items(), columns=['item_code', 'price_count'])
    # data.to_csv("数据/price_count.csv", index=False)
    # plt.plot(map.values())
    # plt.show()
# cout_prices()
def analyse_price_order():
    data=pd.read_csv("数据/price_order.csv")
    map1={}
    map2={}
    for row in data.itertuples():
        # print(row.item_code)
        # print(row.priceorder_list)
        priceoder_list=eval(row.priceorder_list)
        map1[row.item_code]=len(priceoder_list)
        map2[row.item_code]=priceoder_list

    ##map中存储的是
    max_item_code=0
    max_price_count=0
    mapcopy=map1.copy()
    for i in mapcopy.keys():
        if map1[i]<100:
            map1.pop(i)
            map2.pop(i)
        elif map1[i]>max_price_count:
            max_price_count=map1[i]
            max_item_code=i
    paint_list=[]
    for  i in range(5):
        x=random.choice(list(map2.keys()))
        paint_list.append(map2[x])
    paint_price_order(paint_list)

    # print(map)
    # plt.plot(map.values())
    # plt.show()
def paint_price_order(price_list):
    listx=[]
    listy=[]
    for t in price_list:
        temp_x=[]
        temp_y=[]
        for x in t:
            temp_x.append(x[0])
            temp_y.append(x[1])
        result = np.corrcoef(temp_x, temp_y)
        print(result)
        listx.append(temp_x)
        listy.append(temp_y)
    colorlist=["red","blue","green","yellow","black"]
    for i in range(len(listx)):
        plt.plot(listx[i],listy[i],label="item_code"+str(i),color=colorlist[i])
    plt.show()

def count_total_price():
    data=pd.read_csv("数据/order_train1.csv")
    price=data.item_price
    qty=data.ord_qty
    result=np.corrcoef(price,qty)
    print(result)


# analyse_price_order()
# cout_prices()
count_total_price()