import pandas as pd
import numpy as np

a  =  np.array([[684.0, 609.0], [1439.0, 403.0], [1480.0, 561.0], [724.0, 761.0]],dtype=float)
print(a)


data = pd.read_csv("/home/lqy/python_scripy/defect-detecting/many_angle_matching/csvData.csv")

print(data.dtypes)
data[['angle']] = data[['angle']].astype(str)
print(data.dtypes)
#print(data)
#print (data.head(1))
#print(data.columns) #返回全部列名
#print(data.shape)
#打印某行
#print(data.loc[1:2])
#data.loc[2:4, ['point1', 'point2']]    #行中的特定数据
print('9999999999999999999')
#print(data.loc[data['angle']==-15.0]['point1'][0])
print(data['angle'])
match_points = data.loc[data['angle']==str(15.0)]
point = np.array(match_points)

# print(match_points['point1'])
# print(match_points['point1'].replace('\"',('')))
# print( np.array(match_points['point1']))



