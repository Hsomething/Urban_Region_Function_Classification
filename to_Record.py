#将图片和visit数组转换为tfRecord存储
import os
import tensorflow as tf
import cv2
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
def get_data(filename,save_path) :
	"""
	filename: 训练集或验证集列表所保存的TXT文件 如：train.txt  vaild.txt
	save_path：训练集或验证集的npy文件保存的路径  train_visit/npy/
	"""
	with open(filename,'r') as f :
		fileList = f.readlines()
	data = []
	for row in fileList :
		imagepath = row.split('.')[0].split('_')[-1]+'/'+row.split('.')[0]+'.jpg'
		#image = cv2.imread('train_image/after/'+imagepath,cv2.IMREAD_COLOR)
		image = Image.open('data/train_after/'+row.strip())   #图片地址
		#image = Image.open('train_image/after_2/'+imagepath)
		image = np.array(image)
		print(row.strip()+' have read   loading  '+str(fileList.index(row))+'/'+str(len(fileList)))
		visit = np.load(save_path+row.split('.')[0]+'.npy')
		mean = np.mean(visit)
		std = np.std(visit)
		visit = (visit - mean)/std
		label = int(row.strip().split('_')[-1].split('.')[0][-1])-1
		data.append([image,visit,label])
	print('Load file success')
	return data
	
def int64_feature(values):
	if not isinstance(values, (tuple, list)):
		values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
	
	
def to_tfrecord(data,save_path,dataset) :
	'''
	data :list 里面保存着由[图片矩阵，visit数组，label标签]组成的元素
	save_path: 输出的tfRecord保存的路径
	dataset ：保存的tfRecord的数据集的名称
	'''
	outputdir = save_path+'/'+dataset+'.tfrecord'
	writer = tf.python_io.TFRecordWriter(outputdir)
	length = len(data)
	i = 0
	for item in data:
		i +=1
		
		image = item[0].tobytes()
		visit = item[1].tobytes()
		label = item[2]

		example = tf.train.Example(features=tf.train.Features(feature={
			'image': bytes_feature(image),
			'visit': bytes_feature(visit),
			'label': int64_feature(label),
			}))
		
		writer.write(example.SerializeToString())
		sys.stdout.write('\r>> Converting image %d/%d' % (i , length))
		sys.stdout.flush()
	sys.stdout.write('\n')
	sys.stdout.flush()
if __name__ == '__main__' :
	#读取验证集的图片信息，visit数组和label标签
	data = get_data('vaild2.txt','npy/')
	#将data转换为tfRecor
	to_tfrecord(data,'tfRecord','vaild4')
	#读取训练集的图片信息，visit数组和label标签
	data = get_data('train2.txt','npy/')
	#将data转换为为tfRecord
	to_tfrecord(data,'tfRecord','train4')
