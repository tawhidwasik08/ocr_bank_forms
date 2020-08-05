import os
import time
from pathlib import Path
import sys
import argparse

from utility import *

ap = argparse.ArgumentParser()
ap.add_argument("-dir", required=True,help="directory of image folder")
args = vars(ap.parse_args())

# image folder directory
directory = '../'+str(args['dir'])

# get format and variant
f = int(directory.split('/')[2].split('_')[1])
v = int(directory.split('/')[3].split('_')[1])
print('Format:',f,',Variant:',v)


times_req = []
image_count = 0

for filename in os.listdir(directory):

	start_time = time.process_time()
	image_count += 1
	print("Filename:",filename)

	# read image
	image = cv2.imread(directory+filename)

	# image info
	(iH, iW) = image.shape[:2]
	print("Image Res:",iW,'x',iH)
	
	# correct skew
	image = correct_skew(image)

	# different formats have different fields we are looking for//
	# followings are based on observations
	if f == 2 and v != 3:
		# how many fields to extract from image
		desired_field_num = 3

		# crop particular image region
		cropped_image_1 = extract_roi(image,f)

		# convert to grayscale
		gray_1 = cv2.cvtColor(cropped_image_1, cv2.COLOR_BGR2GRAY)
		
		# name of the texts of desired fields
		if v == 1 :
			des_fields_1 = ['Full','Birth','Telephone']
		else:
			des_fields_1 = ['Name','Birth','Tel']

		# get bounding boxes of the texts after ocr
		bb_1 = ocr_roi(gray_1, cropped_image_1, des_fields_1, desired_field_num)

		# mark required fields based on bounding boxes
		image_1 = show_image_field_result(cropped_image_1,bb_1, [x.lower() for x in des_fields_1],f,v)
		v_image = image_1

	else : 
		cropped_image_1 = extract_roi(image, f)
		cropped_image_2 = extract_roi_2(image, f)

		gray_1 = cv2.cvtColor(cropped_image_1, cv2.COLOR_BGR2GRAY)
		gray_2 = cv2.cvtColor(cropped_image_2, cv2.COLOR_BGR2GRAY)

		if f == 1:
			if v == 4:
				desired_field_num = 4
				des_fields_1 = ['Full','Birth','Mobile','Residence']
			elif v == 3:
				desired_field_num = 4
				des_fields_1 = ['Full','Birth','Mobile']
			else:
				desired_field_num = 5
				des_fields_1 = ['Full','Birth','Mobile','ID No','ID','Residence']

		if f == 2 and v == 3:
			desired_field_num = 3
			des_fields_1 = ['Full','Birth','Telephone',]

		bb_1 = ocr_roi(gray_1, cropped_image_1, des_fields_1, desired_field_num)

		des_fields_2 = ['FileID','File/ID','Source']
		bb_2 = ocr_roi(gray_2,cropped_image_2,des_fields_2, 2)

		image_1 = show_image_field_result(cropped_image_1,bb_1, [x.lower() for x in des_fields_1],f,v)
		image_1 = cv2.copyMakeBorder(image_1,10,10,10,10,cv2.BORDER_CONSTANT,value=[50,50,50])

		image_2 = show_image_field_result_2(cropped_image_2,bb_2, ['file','source'],f,v)
		image_2 = cv2.copyMakeBorder(image_2,10,10,10,10,cv2.BORDER_CONSTANT,value=[100,100,100])

		image_1 = imutils.resize(image_1, height=800)
		image_2 = imutils.resize(image_2, height=image_1.shape[:2][0])

		# concatanate two regions two show in one image
		v_image = cv2.hconcat([image_1, image_2])
	
	# save reesult image
	save_dir = '../output_image/format_'+str(f)+'/variant_'+str(v)+'/'
	if not os.path.exists(save_dir):
   		os.makedirs(save_dir)
	cv2.imwrite(save_dir+filename, v_image)

	#show_image(v_image,False)

	times_req.append(time.process_time() - start_time)


print('__________________________________________')
print('Total Images:',image_count)
print('Max time:',max(times_req),'s')
print('Min time:',min(times_req),'s')
print('Avg time:',sum(times_req)/image_count,'s')
print('__________________________________________')