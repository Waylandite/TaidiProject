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
            map[row.item_code]=[]
        if row.item_price not in map[row.item_code]:
            map[row.item_code].append([row.item_price,row.ord_qty])
        else:
            map[row.item_code][map[row.item_code].index(row.item_price)][1]+=row.ord_qty
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
    map={}
    mapindex={}
    for row in data.itertuples():
        # print(row.item_code)
        # print(row.priceorder_list)
        priceoder_list=eval(row.priceorder_list)
        map[row.item_code]=len(priceoder_list)

    ##map中存储的是
    mapcopy=map.copy()
    for i in mapcopy.keys():
        if map[i]<100:
            map.pop(i)
    
    # print(map)
    # plt.plot(map.values())
    # plt.show()
analyse_price_order()