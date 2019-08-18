import os 
import datetime
import numpy as np
import pandas as pd

date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i
	
#生成日期的星期列表和日期列表
for i in range(182):
	date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
	date_int = int(date.__str__().replace("-", ""))
	date2position[date_int] = [i%7, i//7] #星期几，第几周
	datestr2dateint[str(date_int)] = date_int
#将txt文件中的访问数据转换为数组，数组中每个元素代表这个时间点的访问人次
def create_array(lines) :
	array = np.zeros((26,24,7))
	for row in lines :
		temp = []
		for xx in row.split(',') :
			temp.append([xx[0:8],xx[9:].split('|')])
			
		for date ,times in temp :
			x, y = date2position[datestr2dateint[date]]
			for visit in times : 
				# 统计到访的总人数
				array[y][str2int[visit]][x] +=1
	return array
	
#将测试集文visit文件全部转换为以.npy结尾的数组文件	
def array_test() :
	for i in range(0,10000) :
		df = pd.read_csv('test_visit/test/'+str(i).zfill(6)+'.txt',sep = '\t',header = None)
		array = create_array(df[1])
		np.save('test_visit/npy/'+str(i).zfill(6)+'.npy',array)
		print('\r'+str(i).zfill(6)+'.npy have saved	'+str(i)+'/'+'10000')
	print('test_npy end')

#将训练集或验证集的visit文件全部转换为.npy文件		
def array_npy(save_path,txtfile) :  
	'''
	#save_path :npy 文件保存路径     'train_visit/npy/'    'vaild_visit/npy/'
	txtfile : 验证集或测试集列表文件
	'''
	with open(txtfile,'r') as f :
		vaildlist = f.readlines()
	for i in vaildlist :
		df = pd.read_csv('train_visit/train/'+i.strip(),sep = '\t',header = None)
		array = create_array(df[1])
		np.save(save_path+i.split('.')[0]+'.npy',array)
		print('\r'+i.strip()+'have change 	'+str(vaildlist.index(i)+1)+'/'+str(len(vaildlist)))
	print(txtfile.split('.')[0]+'_npy end')
array_npy('npy/','train.txt')
array_npy('npy/','vaild.txt')
array_test()
		
	
	
		
	
