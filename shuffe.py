import os
import random
#from PIL import Image
txtlist = os.listdir('data/train/')


'''for row in txtlist:
	image_path = 'train_image/train/'+row.split('_')[-1].split('.')[0]+'/'+row.split('.')[0]+'.jpg'
	img = Image.open(image_path)
	box = img.getbbox()
	#判断图片是否是全黑还是全白，去除
	extrema = img.convert("L").getextrema()
	if extrema == (0, 0) or extrema == (1, 1):
		pass_list.append(row)
	else :
		if box !=None :
			box = list(box)
			if  box[0] ==0 and box[1] == 0 and box[2] == 100 and box[3] == 100  :
				print(row)
				new_list.append(row)
		else :
			pass_list.append(row)
'''
random.shuffle(txtlist)
random.shuffle(txtlist)
width = int(len(txtlist)*0.2)
print(len(txtlist))

'''with open('pass.txt','w+') as f :
	for i in pass_list:
		f.write(i+'\n')
'''
with open('vaild2.txt','w+') as f :
	for i in range(0,width) :
		f.write(txtlist[i]+'\n')

with open('train2.txt','w+') as f :
	for i in range(width,len(txtlist)) :
		f.write(txtlist[i]+'\n')