import tensorflow as tf
from mymodel import *
import numpy as np
import time

batch_size = 256
learning = 1e-3
epochs = 10
def read_trainSet(filename) :
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
		features = {
					'image' :tf.FixedLenFeature([], tf.string),
					'visit' :tf.FixedLenFeature([], tf.string),
					'label' :tf.FixedLenFeature([], tf.int64),})
	image = tf.decode_raw(features['image'],tf.uint8)
	image = tf.reshape(image,[100,100,3])
	#image = tf.random_crop(image,[88,88,3])
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_flip_up_down(image)
	image = tf.image.random_brightness(image,max_delta=30)
	image = tf.image.random_saturation(image,lower=0.2,upper=1.8)
	image = tf.cast(image,tf.float32)
	image = tf.image.per_image_standardization(image)
	
	visit = tf.decode_raw(features['visit'],tf.float64)
	visit = tf.reshape(visit,[26,24,7])
	label = tf.cast(features['label'], tf.int64)
	label = tf.one_hot(indices = label,depth = 9,name = 'one_hot')

	return image,visit,label


def read_vaildSet(filename) :
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features = {
																	'image':tf.FixedLenFeature([],tf.string),
																	'visit':tf.FixedLenFeature([],tf.string),
																	'label':tf.FixedLenFeature([],tf.int64),})
	image = tf.decode_raw(features['image'],tf.uint8)
	image = tf.reshape(image,[100,100,3])
	#image = tf.random_crop(image,[88,88,3])
	image = tf.cast(image,tf.float32)
	image = tf.image.per_image_standardization(image)
	
	visit = tf.decode_raw(features['visit'],tf.float64)
	visit = tf.reshape(visit,[26,24,7])
	label = tf.cast(features['label'], tf.int64)
	label = tf.one_hot(indices = label,depth = 9,name = 'one_hot')
	return image,visit,label
def load_trainingSet() :
	with tf.name_scope('input_train') :
		image_train,visit_train,label_train = read_trainSet('tfrecord/train3.tfrecord')
		image_batch,visit_batch,label_batch = tf.train.shuffle_batch([image_train,visit_train,label_train],
					batch_size = batch_size,
					capacity = 2048,
					min_after_dequeue = 2000,
					num_threads = 4)
		return image_batch,visit_batch,label_batch
		
def load_vaildSet() :
	with tf.name_scope('input_vaild') :
		image_vaild,visit_vaild,label_vaild = read_vaildSet('tfrecord/vaild3.tfrecord')
		image_batch,visit_batch,label_batch = tf.train.shuffle_batch(
				[image_vaild,visit_vaild,label_vaild],batch_size = batch_size,capacity = 2048,min_after_dequeue = 2000,num_threads = 4)
		return image_batch,visit_batch,label_batch
def train():
	#model = load_model('model/class_3_60.h5')
	model = my_net().ClassiFilerNet()
	image_vaild_batch,visit_vaild_batch,label_vaild_batch = load_vaildSet()
	image_train_batch,visit_train_batch,label_train_batch = load_trainingSet()
	sess = tf.Session()
	coord = tf.train.Coordinator()
	thread = tf.train.start_queue_runners(sess = sess,coord = coord)
	model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',
				metrics=['accuracy'])
	for i in range(epochs) :
		for j in range(int(31784/batch_size)+1) :
			time1 = time.time()
			image_train,visit_train,label_train = sess.run(
				[image_train_batch,visit_train_batch,label_train_batch])
			hist = model.train_on_batch(x = [image_train,visit_train],y = label_train,class_weight = 'auto')
			print('epoch:%d step: %d/%d train loss=%0.3f ,acc = %0.3f time=%0.3f'%(i,j,125,float(hist[0]),float(hist[-1]),time.time()-time1))
	
			if j%10 == 0:
				image_vaild,visit_vaild,label_vaild = sess.run(
					[image_vaild_batch,visit_vaild_batch,label_vaild_batch])
				result = model.evaluate(x = [image_vaild,visit_vaild],y = label_vaild,batch_size = batch_size,verbose = 1)
				print('valid loss= %0.3f,acc = %0.3f time=%0.3f:'%(result[0],result[-1],time.time()-time1))
			if j%20 ==0 and j!=0 :
				model.save('model/class_'+str(i)+'_'+str(j)+'.h5')
				print('模型保存成功')
	coord.request_stop()
	coord.join(threads)		
if __name__ == '__main__' :
	train()