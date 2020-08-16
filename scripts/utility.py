import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import imutils
from fuzzywuzzy import fuzz
import math
from shapely.geometry import Polygon

# function to show any image to check out 
def show_image(image,resized):
	if (resized == True):
		image = imutils.resize(image, height=800)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return None

def extract_roi(image, roi, ratio_val):
	width = image.shape [1]
	height = image.shape [0]

	x = int(width*ratio_val['xr'])
	y = int(height*ratio_val['yr'])
	w = int(width*ratio_val['wr'])
	h = int(height*ratio_val['hr'])

	crop_img = None
	if roi == 'roi_top_left':
		crop_img = image[y:y+h, x:x+w]
	elif roi == 'roi_bottom_right':
		crop_img = image[y-h:y, x-w:x]
	elif roi == 'roi_top_right':
		crop_img = image[y:y+h, x-w:x]
	elif roi == 'roi_bottom_left':
		crop_img = image[y:y-h, x:x+w]
	return crop_img


# deskewing image as much as possible //
# incase of failure keeping as is
def correct_skew(image):
	image_copy = image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# padding
	black = [0,0,0]
	gray = cv2.copyMakeBorder(gray,50,50,50,50,cv2.BORDER_CONSTANT,value=black)
	gray_2 = gray
	
	edged = cv2.Canny(gray,50,200)
	kernel = np.ones((13, 13), np.uint8)
	filtered = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

	# find contours
	cnts, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
	
	image = cv2.cvtColor(filtered,cv2.COLOR_GRAY2RGB)

	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

	# approximate from top 10 contours with 4 cornered one
	for c in cnts:
		peri = cv2.arcLength(c,True)
		approx = cv2.approxPolyDP(c,0.015 * peri,True)

		if len(approx) == 4 :
			screenCnt= approx
			break

	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 20)

	rect = cv2.minAreaRect(screenCnt)
	center = rect[0]
	angle = rect[2]
	(h, w) = image_copy.shape[:2]
	center = (w // 2, h // 2)
	if (angle < -45):
		M = cv2.getRotationMatrix2D(center, 90+angle, 1.0)
	else:
		M = cv2.getRotationMatrix2D(center, angle, 1.0)

	image = cv2.warpAffine(image_copy, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return image

# different sets of preprocessing techniques
def image_smoothening(image):
	BINARY_THRESHOLD = 150
	ret1, th1 = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
	ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	blur = cv2.GaussianBlur(th2, (1, 1), 0)
	ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return th3


def remove_noise_and_smooth(image):
	filtered = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,3)
	kernel = np.ones((1, 1), np.uint8)
	opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	image = image_smoothening(image)
	or_image = cv2.bitwise_or(image, closing)
	return or_image


def preprocess_1(image):
	kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	image = cv2.filter2D(image, -1, kernel)
	image = remove_noise_and_smooth(image)	
	return image


def preprocess_2(image):
	kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	image = cv2.filter2D(image, -1, kernel)
	return image


def preprocess_3(image):
	image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	image = cv2.filter2D(image, -1, kernel)
	return image


def preprocess_4(image):
	image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	return image

# partial text checking among desired fields
def check_field(text,des_fields):
	for item in des_fields:
			if fuzz.ratio(item,text) > 73:
				return True
	return False

# ocr one particular image and return their bounding boxes in a dictionary
def ocr(image,original,des_fields):
	# --psm 11 tries to find as much as text possible
	custom_config = r'--oem 3 --psm 11'
	details = pytesseract.image_to_data(image, output_type= Output.DICT, config=custom_config, lang='eng')
	(h1, w1) = original.shape[:2]
	(h2, w2) = image.shape[:2]
	temp_list = []
	total_boxes = len(details['text'])
	for sequence_number in range(total_boxes):
		if int(details['conf'][sequence_number]) >= 20:
			if check_field(details['text'][sequence_number],des_fields):
				temp_dict = {}
				(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
				x = int((w1 * x) / w2)
				y = int((h1 * y) / h2)
				w = int((w1 * w) / w2)
				h = int((h1 * h) / h2)
				temp_dict[details['text'][sequence_number]] = (x,y,w,h)
				temp_list.append(temp_dict)
				font = cv2.FONT_HERSHEY_SIMPLEX
	return temp_list


# bounding boxes keys are original ocr-ed text //
# clean up the text to be matched with the closest with desired ones
def clean_bound_box(bound_boxes,des_fields,fvr_data):
	
	temp_list = []
	temp_list_2 = []
	
	try:
		duplicates = fvr_data['duplicates']
	except KeyError:
		duplicates = []

	# clean partial text in bounding box dictionary to be matched with desired field texts
	for item in bound_boxes:
		max_ratio = 0
		max_matched_text = ''
		for key,val in item.items():
			for d in des_fields:
				if fuzz.ratio(key.lower(),d.lower()) > max_ratio:
					max_ratio = fuzz.ratio(key.lower(),d.lower())
					max_matched_text = d

			if max_matched_text not in [k for d in temp_list_2 for k in d.keys()] or max_matched_text in duplicates:
				temp_list.append({max_matched_text:val})
				temp_list_2.append({max_matched_text:max_ratio})
			else :
				existent_fuzz_ratio = [ b[max_matched_text] for b in temp_list_2 if max_matched_text in b][0]
				if existent_fuzz_ratio < max_ratio:
					temp_list = [{key: value for key, value in dict.items() if key != max_matched_text} for dict in temp_list]
					temp_list_2 = [{key: value for key, value in dict.items() if key != max_matched_text} for dict in temp_list_2]
					temp_list.append({max_matched_text:val})
					temp_list_2.append({max_matched_text:max_ratio})

	# if swap is not need for a field, remove unnecessary ones
	temp_list = list(filter(None, temp_list))					
	bound_boxes_items = list(k for d in temp_list for k in d.keys())
	for b in temp_list:
		key = list(b.keys())[0]
		try:
			if fvr_data['swap_fields'][key] in bound_boxes_items:
				temp_list.remove(b)
		except KeyError:
			pass

	return temp_list


def draw_box(image,key,x,y,w,h,color):
	font = cv2.FONT_HERSHEY_SIMPLEX
	image = cv2.putText(image,key,(x, y), font, 0.8, color, 2, cv2.LINE_AA)
	image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	return image

# find co ordinates of fields based on format,variant and text
# for top left portion
def show_image_field_result(image, bound_boxes, fvr_data):
	try:
		des_fields = list(fvr_data['des_fields'].keys()) \
					+list(fvr_data['des_fields_extra'].keys())
	except KeyError:
		des_fields = des_fields = list(fvr_data['des_fields'].keys())

	ref = fvr_data['ref']
	def_ref_dis = fvr_data['default_dis_x']

	bound_boxes = clean_bound_box(bound_boxes,des_fields,fvr_data)

	# calculate mid points for the reference boxes // 
	# in case of missing kept a deafult avg value
	try:
		ref_x1 = [d[ref[0]] for d in bound_boxes if ref[0] in d][0][0]
		ref_w1 = [d[ref[0]] for d in bound_boxes if ref[0] in d][0][2]
		ref_x2 = [d[ref[1]] for d in bound_boxes if ref[1] in d][0][0]
		ref_w2 = [d[ref[1]] for d in bound_boxes if ref[1] in d][0][2]

		ref_x1_mid = int(((2*ref_x1)+ref_w1)/2)	
		ref_x2_mid = int(((2*ref_x2)+ref_w2)/2)

		# get difference
		dif = abs(ref_x2_mid-ref_x1_mid)
	except :
		dif = def_ref_dis

	(imgH, imgW) = image.shape[:2]

	for b in bound_boxes:
		for key,val in b.items():
			
			# get mid point of the box
			mid_x = int (((2*val[0])+val[2])/2)
			mid_y = int (((2*val[1])+val[3])/2)

			# try to get ratio values for each field //
			# if missing try to get from swappable field
			try:
				r_vals = fvr_data['des_fields'][key]
			except KeyError:
				r_vals = fvr_data['des_fields_extra'][key]
				key = fvr_data['swap_fields'][key]

			# calculate offset
			x = int (mid_x+dif*r_vals[0])
			y = int (mid_y-dif*r_vals[1])
			if r_vals[2] == None:
				w = imgW-x
			else:
				w = int (dif*r_vals[2])
			h = int(dif*r_vals[3])

			image = draw_box (image, key, val[0], val[1], val[2], val[3], (25, 25, 255))
			image = draw_box(image, key, x, y, w, h, (255, 25, 0))

	return image


# check if bounding boxes are overlapped 
def overlap(rect1,rect2):
	x,y,w1,h1 = rect1
	a,b,w2,h2 = rect2
	p1 = Polygon([[x, y], [x, y+h1], [x+w1, y], [x+w1, y+h1]])
	p2 = Polygon([[a, b], [a, b+h2], [a+w2, b], [a+w2, b+h2]])
	return(p1.intersects(p2))


# remove duplicate bounding boxes
def remove_duplicates(bb):
	remove_index = []

	for i in range(len(bb)):
		for k1,v1 in bb[i].items():
			rectangle_1 = v1

		for j in range(i+1,len(bb)):
			if j != i:
				for k2,v2 in bb[j].items():
					rectangle_2 = v2
				if overlap(rectangle_1,rectangle_2):
					remove_index.append(j)
	bb = [j for i, j in enumerate(bb) if i not in remove_index]
	return bb


def patterns_from_bb(bb):
	bb = remove_duplicates(bb)
	return len(bb), bb


def ocr_roi(gray, cropped_image, des_fields, des_pattern_num):
	# ocr on base grayscale image and get bounding boxes of the found texts
	bb = ocr(gray,cropped_image,des_fields)  

	# no. of patterns from bounding boxes after duplicate/overlapped boxes removed
	pattern_count, bb = patterns_from_bb(bb)

	# list of preprocessings to run until all not patterns are found
	preprocess_list = [preprocess_1(gray),preprocess_2(gray),preprocess_3(gray),preprocess_4(gray)]

	i = 0
	while (i < len(preprocess_list) and pattern_count < des_pattern_num):
		preprocessed_image = preprocess_list[i]
		bb = bb + ocr(preprocessed_image, cropped_image, des_fields)
		pattern_count, bb = patterns_from_bb(bb)
		i += 1

	return bb

