import os
import time
import sys
import argparse
import json

from utility import *

ap = argparse.ArgumentParser()
ap.add_argument("-dir", required=True,help="directory of image folder")
args = vars(ap.parse_args())

# image folder directory
path = os.getcwd() 
parent = os.path.join(path, os.pardir)
directory = os.path.abspath(parent)+'/'+str(args['dir'])

# get format and variant
f = int(directory.split('/')[2].split('_')[1])
v = int(directory.split('/')[3].split('_')[1])
print('Format:',f,', Variant:',v)

# load format and variant config from json file to dictionary
with open('config.json') as json_file:
		config_data = json.load(json_file)

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

	# get config information by region
	fv_data = config_data['nbl']['format_'+str(f)]['variant_'+str(v)]
	rois = list(fv_data.keys())

	image_roi_list = [] 

	for roi in rois:

		ratio_val = fv_data[roi]['ratios']

		# crop image and convert to grayscale
		cropped_image = extract_roi(image, roi, ratio_val)
		gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

		# how many and name of the fields 
		des_fields_num = fv_data[roi]['des_field_num']
		try:
			des_fields = list(fv_data[roi]['des_fields'].keys()) \
						+list(fv_data[roi]['des_fields_extra'].keys())
		except KeyError:
			des_fields = list(fv_data[roi]['des_fields'].keys())

		# get ocr'd texts bounding boxes
		bb = ocr_roi(gray, cropped_image, des_fields, des_fields_num)

		# mark desired boxes in image
		result_image = show_image_field_result(cropped_image, bb, fv_data[roi])

		result_image = imutils.resize(result_image, height=800)

		image_roi_list.append(result_image)

		print('---------\n\n')

	v_image = cv2.hconcat(image_roi_list)

	# save result image
	save_dir = parent+'/output_image/format_'+str(f)+'/variant_'+str(v)+'/'
	if not os.path.exists(save_dir):
   		os.makedirs(save_dir)
	cv2.imwrite(save_dir+filename, v_image)

	times_req.append(time.process_time() - start_time)


print('__________________________________________')
print('Total Images:',image_count)
print('Max time:',max(times_req),'s')
print('Min time:',min(times_req),'s')
print('Avg time:',sum(times_req)/image_count,'s')
print('__________________________________________')