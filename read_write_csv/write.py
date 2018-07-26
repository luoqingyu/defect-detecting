import csv
csvFile = open("csvData.csv", "w")            #创建csv文件
writer = csv.writer(csvFile)                  #创建写的对象
#先写入columns_name
writer.writerow(["index","a_name","b_name"])     #写入列的名称
#写入多行用writerows                                #写入多行
writer.writerows([[1,'a','b'],[2,'c','d'],[3,'d','e']])
csvFile.close()



# import csv
# import numpy  as np
# import math
# #计算一个点旋转后的映射位置
# def get_romate_point(point,center,angle):
#     new_point = []
#     distence = []
#     distence.append(point[0] - center[0])
#     distence.append(point[1] - center[1])
#     new_point.append( round(distence[0]* math.cos(angle) + distence[1] * math.sin(angle) + center[0]))
#     new_point.append( round(-distence[0] * math.sin(angle) + distence[1]* math.cos(angle) + center[1]))
#     return new_point
#
# def get_many_romate_point(Pts,w_img , h_img,i):
#     four_point_str=''
#     new_pts = []
#     #four_point_str = '_' + str(i) + '_'
#     for row in Pts:
#         point = get_romate_point(row, (w_img / 2, h_img / 2), i * math.pi / 180)
#         new_pts.append(point)
#     return  new_pts
# #python2可以用file替代open
#
#
#
#
# # 读取csv文件方式1
# csvFile = open("csvData.csv", "a+")
# Pts = np.array([[799, 463], [1581, 460], [1580, 623], [798, 620]], np.int32)
# Pts_si = np.array([[984, 463], [1024, 463], [1024, 504], [984, 504]], np.int32)
# str_four_point_pts = get_many_romate_point(Pts,1000, 1000, 5)
# #Pts = [[799, 463], [1581, 460], [1580, 623], [798, 620]]
#
# writer = csv.writer(csvFile)
# #先写入columns_name
# #writer.writerow(["index","a_name","b_name"])
# #写入多行用writerows
# writer.writerows([[str_four_point_pts,1,2],[[1,2],2,3],[[2,3],3,4]])
# csvFile.close()



# with open("test.csv","w") as csvfile:
#     Pts = np.array([[799, 463], [1581, 460], [1580, 623], [798, 620]], np.int32)
#     Pts_si = np.array([[984, 463], [1024, 463], [1024, 504], [984, 504]], np.int32)
#     str_four_point_pts = get_many_romate_point(Pts,1000, 1000, 5)
#     #Pts = [[799, 463], [1581, 460], [1580, 623], [798, 620]]
#     writer = csv.writer(csvfile)
#     #先写入columns_name
#     writer.writerow(["index","a_name","b_name"])
#     #写入多行用writerows
#     writer.writerows([[str_four_point_pts,1,2],[[1,2],2,3],[[2,3],3,4]])