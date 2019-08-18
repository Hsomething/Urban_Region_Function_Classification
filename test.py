from model import FunctionModel
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

graph = tf.Graph()
sess = tf.InteractiveSession(graph = graph)
def ComputeHist(img):
    h,w = img.shape
    hist, bin_edge = np.histogram(img.reshape(1,w*h), bins=list(range(257)))
    return hist
    
def ComputeMinLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if (sum >= (pnum * rate * 0.01)):
            return i
            
def ComputeMaxLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255-i]
        if (sum >= (pnum * rate * 0.01)):
            return 255-i
            
def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):    #获取阈值外的像素值 i< minlevel，i> maxlevel
            if (i < minlevel):
                newmap[i] = 0
            elif (i > maxlevel):
                newmap[i] = 255
            else:
                newmap[i] = (i-minlevel)/(maxlevel-minlevel) * 255
        return newmap
#图片的自适应色阶去雾气        
def CreateNewImg(img):
    h,w,d = img.shape
    newimg = np.zeros([h,w,d])
    for i in range(d):
        imgmin = np.min(img[:,:,i])
        imgmax = np.max(img[:,:,i])
        imghist = ComputeHist(img[:,:,i])
        minlevel = ComputeMinLevel(imghist, 8.3, h*w)
        maxlevel = ComputeMaxLevel(imghist, 2.2, h*w)
        newmap = LinearMap(minlevel,maxlevel)
        if (len(newmap) ==0 ):
            continue
        for j in range(h):
            newimg[j,:,i] = newmap[img[j,:, i]]
    return newimg
with sess.graph.as_default() :
	with sess.as_default() :
		model = FunctionModel()
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		var_list = [var for var in tf.global_variables() if "moving" in var.name]
		var_list += [var for var in tf.global_variables() if "global_step" in var.name ]
		var_list += tf.trainable_variables()
		saver = tf.train.Saver(var_list = var_list,max_to_keep = 1)
		last_file = tf.train.latest_checkpoint('model/')
		if last_file :
			tf.logging.info('Restoring model from {}'.format(last_file))
			saver.restore(sess,last_file)
images = []
visits = []
for i in range(10000) :
	
	image = cv2.imread('test_image/test/'+str(i).zfill(6)+".jpg",cv2.IMREAD_COLOR)
	image = CreateNewImg(image).astype('uint8')
	image = (image-np.mean(image))/np.std(image)
	image = image[0:88,0:88,:]
	visit = np.load('test_visit/npy/'+str(i).zfill(6)+'.npy')
	visit = (visit-np.mean(visit))/np.std(visit)
	print('load data '+str(i)+'/10000')
	images.append(image)
	visits.append(visit)

predictions = []

for i in range(10) :
	predictions.extend(sess.run(tf.argmax(model.prediction,1),
						feed_dict={model.image: images[i*1000:i*1000+1000],
                                     model.visit: visits[i*1000:i*1000+1000],
                                     model.training: False}))
	print(i)

if not os.path.exists('result/') :
		os.mkdir('result/')
with open('result/result.txt','w+') as f :
	for index,prediction in enumerate(predictions) :
		f.write("%s \t %03d\n"%(str(index).zfill(6), prediction+1))
print('测试完成')
