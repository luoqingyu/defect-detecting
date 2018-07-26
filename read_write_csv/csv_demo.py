# encoding: utf-8
import pandas as pd
import numpy as np
df=pd.read_csv('./data.csv') #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
#print (df.head(1))          #返回多少行
#print (df.tail())
#作为示例，输出CSV文件的前5行和最后5行，这是pandas默认的输出5行，可以根据需要自己设定输出几行的值
print(df.columns) #返回全部列名
dimensison = df.shape        #返回数据的格式，数组，（行数，列数）
df.values     #返回底层的numpy数据