import pandas as pd
import numpy as np
data = pd.read_csv("/home/lqy/python_scripy/defect-detecting/many_angle_matching/csvData.csv")
df =data
data[['angle']] = data[['angle']].astype(str)
print(data.shape)                            #查看数据形状
print(data.info())
print(data.dtypes)
print(data['angle'].dtype)
#print(df.isnull())
#print(df['angle'].unique())
#print(df.values )
#print(df.columns[5])
#df.head() #默认前10行数据
#df.tail()    #默认后10 行数据
match_points = data.loc[data['angle']==str(15.0)]
#z = data.iloc[:1,:2]
z =  match_points.iloc[0,1:].values
for point_num in range(int(z.shape[0]/8)):
    point = z[point_num*8:point_num*8+8]
    print(point)
    old_point = []
    for i in range(4):
        old_point.append([int(point[i * 2]) - 708 , int(point[i * 2 + 1]) - 333])
    print(old_point)




#for i in len(z)/8
#print(z.iloc[2])

#old_point = []
#for i in range(4):
#    old_point.append([int(z[0,i * 2]) - 708 , int(z[0,i * 2 + 1]) - 333 ])

#print(z)

# print(data.dtypes)
# data[['angle']] = data[['angle']].astype(str)
# print(data.dtypes)
#print(data)
#print (data.head(1))
#print(data.columns) #返回全部列名
#print(data.shape)
#打印某行
#print(data.loc[1:2])
#data.loc[2:4, ['point1', 'point2']]    #行中的特定数据
# print('9999999999999999999')
# #print(data.loc[data['angle']==-15.0]['point1'][0])
# print(data['angle'])
# match_points = data.loc[data['angle']==str(15.0)]
# point = np.array(match_points)

# print(match_points['point1'])
# print(match_points['point1'].replace('\"',('')))
# print( np.array(match_points['point1']))



