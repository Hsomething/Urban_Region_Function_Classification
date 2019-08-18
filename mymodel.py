from keras.utils import plot_model
import keras.applications.resnet50 as resnet
from keras.layers import Dense,Flatten,concatenate,Activation, Dropout, Conv2D, MaxPool2D,AvgPool2D,BatchNormalization,SeparableConv2D
from keras.models import Model,load_model
from keras.layers import Input


class my_net(object) :
	def Resnet(self) :
		#path='resnet50_weights_tf_dim_ordering_tf_kernels.h5'
		#res = resnet.ResNet50(weights=path,include_top=True,input_shape=(100,100,3))
		
		res = load_model('model_image/class_6_19.h5') #4_18
		print('模型读取成功')
		for layer in res.layers:
			layer.trainable = False
		#for layer in res.layers [41:50]:
			#layer.trainable = True
		x = res.layers[-2].output
		#x = Dense(256,activation = 'relu')(x)
		model = Model(inputs = res.input,outputs = x)
		return model
		
	'''def resnet34(self) :
		inpt = Input(shape=(26,24,7))
		
		x = self.Conv2d_BN(inpt,nb_filter=64,kernel_size=(3,3),strides=(2,2),padding='valid')
		x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
		x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))
		x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))
		x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))
		x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))
		x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))
		#(7,7,512)
		x = self.Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
		x = self.Conv_Block(x,nb_filter=128,kernel_size=(3,3))
		x = self.Conv_Block(x,nb_filter=128,kernel_size=(3,3))
		x = AveragePooling2D(pool_size=(7,7))(x)
		x = Flatten()(x)
		model = Model(inputs = inpt,outputs = x)
		return class_model
		'''
	def LeNet(self):
		inpt = Input(shape = (26,24,7))
		x = Conv2D(32,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform')(inpt)
		x = AvgPool2D(pool_size=(2,2))(x)
		x = Conv2D(64,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform')(x)
		x = AvgPool2D(pool_size=(2,2))(x)
		x = Conv2D(128,(2,2),strides=(1,1),padding = 'valid',activation = 'relu')(x)
		x = MaxPool2D(pool_size = (1,1))(x)
		x = Flatten()(x)
		x = Dense(256,activation = 'relu')(x)
		class_model = Model(inputs = inpt,outputs = x)
		return class_model
	'''def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
		if name is not None:
			bn_name = name + '_bn'
			conv_name = name + '_conv'
		else:
			bn_name = None
			conv_name = None
 
		x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
		x = BatchNormalization(axis=3,name=bn_name)(x)
		return x
 
	def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
		x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
		x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
		if with_conv_shortcut:
			shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
			x = add([x,shortcut])
			return x
		else:
			x = add([x,inpt])
			return x
	'''
	def ClassiFilerNet(self):  
		input1 = self.Resnet()
		input2 = self.LeNet()                     # 孪生网络中的另一个特征提取
		for layer in input2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
			layer.name = layer.name + str("_2")
		inp1 = input1.input
		inp2 = input2.input
		merge_layers = concatenate([input1.output, input2.output],axis = 1)        # 进行融合，使用的是默认的sum，即简单的相加
		#fc1 = Dense(256, activation='relu',name ='fc1')(merge_layers)
		fc2 = Dense(9, activation='softmax',name='fc2')(merge_layers)

		class_models = Model(inputs=[inp1, inp2], outputs=fc2)
		return class_models
if __name__ == '__main__' :
	model = my_net().ClassiFilerNet()
	model.summary()



