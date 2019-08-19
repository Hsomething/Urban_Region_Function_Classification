# Urban_Region_Function_Classification
百度点石平台一带一路国际大数据竞赛模型源码

数据来源：百度点石平台一带一路大数据竞赛

通过给定的遥感图像和用户访问数据对该图片下的地点进行分类


id与类别：
	
	CategoryID	Functions of Areas

	001	        Residential area

	002	        School

	003	        Industrial park

	004	        Railway station

	005	        Airport

	006	        Park

	007	        Shopping area

	008	        Administrative district

	009	        Hospital

此模型基于python+keras,opencv,numpy,tensorflow实现

模型是keras构建的孪生网络模型，孪生网络的一边是通过resnet50实现对图像的特征提取，另一边是通过LeNet对用户访问数据进行特征提取，最后在全连接层特征融合最后再做分类
