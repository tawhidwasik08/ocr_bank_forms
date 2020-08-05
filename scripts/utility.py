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

# crop out top left portion
def extract_roi(image,f):
	width = image.shape [1]
	height = image.shape [0]

	if (f==1):
		x = int(width*0.015)
		y = int(height*0.2)
		w = int(width*0.5)
		h = int(height*0.38)
	else:
		x = int(width*0.015)
		y = int(height*0.12)
		w = int(width*0.5)
		h = int(height*0.25)
	crop_img = image[y:y+h, x:x+w]
	return crop_img

# crop out bottom right portion
def extract_roi_2(image,f):
	width = image.shape [1]
	height = image.shape [0]

	if f == 1:
		x = int(width)
		y = int(height)
		w = int(width*0.5)
		h = int(height*0.25)
	else:
		x = int(width)
		y = int(height)
		w = int(width*0.5)
		h = int(height*0.30)
	crop_img = image[y-h:y, x-w:x]
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


# def show_image_result(image,bound_boxes):
# 	for item in bound_boxes:
# 		for key,val in item.items():
# 			(x, y, w, h) = val
# 			font = cv2.FONT_HERSHEY_SIMPLEX
# 			image = cv2.putText(image,key,(x, y), font, 0.8, (255, 25, 0), 2, cv2.LINE_AA)
# 			image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 	show_image(image,True)

# bounding boxes keys are original ocr-ed text //
# clean up the text to be matched with the closest with desired ones
def clean_bound_box(bound_boxes,des_fields):
	temp_list = []
	temp_list_2 = []
	for item in bound_boxes:
		max_ratio = 0
		max_matched_text = ''
		for key,val in item.items():
			for d in des_fields:
				if fuzz.ratio(key.lower(),d) > max_ratio:
					max_ratio = fuzz.ratio(key.lower(),d)
					max_matched_text = d

			if max_matched_text not in [k for d in temp_list_2 for k in d.keys()] or max_matched_text=='mobile':
				temp_list.append({max_matched_text:val})
				temp_list_2.append({max_matched_text:max_ratio})
			else :
				existent_fuzz_ratio = [ b[max_matched_text] for b in temp_list_2 if max_matched_text in b][0]
				if existent_fuzz_ratio < max_ratio:
					temp_list = [{key: value for key, value in dict.items() if key != max_matched_text} for dict in temp_list]
					temp_list_2 = [{key: value for key, value in dict.items() if key != max_matched_text} for dict in temp_list_2]
					temp_list.append({max_matched_text:val})
					temp_list_2.append({max_matched_text:max_ratio})

	return temp_list


def draw_result_box(image,key,x,y,w,h):
	font = cv2.FONT_HERSHEY_SIMPLEX
	image = cv2.putText(image,key,(x, y), font, 0.8, (255, 25, 0), 2, cv2.LINE_AA)
	image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 50, 0), 2)
	return image

# find co ordinates of fields based on format,variant and text
# for top left portion
def show_image_field_result(image,bound_boxes,des_fields,f,v):
	bound_boxes = clean_bound_box(bound_boxes,des_fields)

	for item in bound_boxes:
		for key,val in item.items():
			if key == 'birth':
				birth_box = val
			if key == 'full' or key == 'name':
				full_box = val

	try:
		x_dif = int (((birth_box[0]+birth_box[0]+birth_box[2])/2)- ((full_box[0]+full_box[0]+full_box[2])/2))
		y_dif = int (((birth_box[1]+birth_box[1]+birth_box[3])/2)- ((full_box[1]+full_box[1]+full_box[3])/2))
	except UnboundLocalError:
		if f == 1:
			x_dif = 505
			y_dif = 140
		if f == 2 :
			x_dif = 420
			y_dif = 75

	if f == 1:
		x_rat_full, y_rat_full, h_rat_full = (0.25, 0.2, 0.3)
		x_rat_id, y_rat_id, h_rat_id = (0.1, 0.22, 0.3)
		x_rat_birth, y_rat_birth, h_rat_birth = (0.08, 0.125, 0.3)
		x_rat_mobile, y_rat_mobile, h_rat_mobile = (0.08,0.18, 0.28)
	
	if f == 2:
		if v == 1 or v ==3:
			x_rat_full, y_rat_full, h_rat_full = (0.15, 0.35, 0.5)
			x_rat_birth, y_rat_birth, h_rat_birth = (0.07, 0.25, 0.5)
			x_rat_tp, y_rat_tp, h_rat_tp = (0.1,0.35, 0.45)

		if v == 2:
			x_rat_full, y_rat_full, h_rat_full = (0.07, 0.25, 0.3)
			x_rat_birth, y_rat_birth, h_rat_birth = (0.07, 0.25, 0.3)
			x_rat_tp, y_rat_tp, h_rat_tp = (0.1,0.25, 0.3)
	
	(ih, iw) = image.shape[:2]
	for item in bound_boxes:
		for key,val in item.items():
			(x, y, w, h) = val
			image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,15, 255), 2)
			if (key == 'full' or key =='name'):
				x = int (((x+x+w)/2) + x_dif*x_rat_full)
				y = int (((y+y+h)/2) - y_dif*y_rat_full)
				w = int (iw-x)
				h = int(y_dif*h_rat_full)
				image = draw_result_box(image,key, x,y,w,h)

			if (key == 'id'):
				x = int (((x+x+w)/2) + x_dif*x_rat_id)
				y = int (((y+y+h)/2) - y_dif*y_rat_id)
				w = int (iw-x)
				h = int(y_dif*h_rat_id)
				image = draw_result_box(image,key, x,y,w,h)

			if (key == 'birth'):
				x = int (((x+x+w)/2) + x_dif*x_rat_birth)
				y = int (((y+y+h)/2) - y_dif*y_rat_birth)
				w = int (iw-x)
				h = int(y_dif * h_rat_birth)
				image = draw_result_box(image,key, x,y,w,h)

			if (key == 'telephone' or key == 'tel'):
				x = int (((x+x+w)/2) + x_dif*x_rat_tp)
				y = int (((y+y+h)/2) - y_dif*y_rat_tp)
				w = int (iw-x)
				h = int(y_dif * h_rat_tp)
				image = draw_result_box(image,key, x,y,w,h)

			if (key == 'mobile'):
				x = int (((x+x+w)/2) + x_dif*x_rat_mobile)
				y = int (((y+y+h)/2) - y_dif*y_rat_mobile)
				if f == 1 and v ==3:
					w = int(x_dif * 0.75)
				else:
					w = int (iw-x)
				h = int(y_dif * h_rat_mobile)
				image = draw_result_box(image,key, x,y,w,h)

			if not any('mobile' in d for d in bound_boxes):
				if (key == 'residence'):
					x = int (((x+x+w)/2) + x_dif)
					y = int (((y+y+h)/2) - y_dif*y_rat_mobile)
					w = int (iw-x)
					h = int(y_dif * h_rat_mobile)
					image = draw_result_box(image,key, x,y,w,h)
	return image

# find co ordinates of fields based on format,variant and text
# for bottom right portion
def show_image_field_result_2(image,bound_boxes, des_fields,f,v):	
	bound_boxes = clean_bound_box(bound_boxes,des_fields)

	for item in bound_boxes:
		for key,val in item.items():
			if key == 'file':
				file_box = val
			if key == 'source':
				source_box = val

	try:
		x_dif = int (((source_box[0]+source_box[0]+source_box[2])/2)- ((file_box[0]+file_box[0]+file_box[2])/2))
		y_dif = int (((file_box[1]+file_box[1]+file_box[3])/2)- ((source_box[1]+source_box[1]+source_box[3])/2))
	except UnboundLocalError:
		if f == 1 and v == 3:
			x_dif = 520
			y_dif = 65
		else :
			x_dif = 510
			y_dif = 5
	
	(ih, iw) = image.shape[:2]
	for item in bound_boxes:
		for key,val in item.items():
			(x, y, w, h) = val
			image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,15, 255), 2)
			if (key == 'file'):
				x = int (((x+x+w)/2) + w)
				y = int (((y+y+h)/2)*0.90)
				w = int (source_box[0]-x)
				if f == 1 and v == 3:
					h = int(x_dif * 0.15)
				else:
					h = int(y_dif)
				image = draw_result_box(image,key, x,y,w,h)
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


def ocr_roi(gray,cropped_image,des_fields,p):
	bb = ocr(gray,cropped_image,des_fields)  
	# no. of patterns from bounding boxes
	pattern_count,bb = patterns_from_bb(bb)

	# if all not patterns are found, serialize next preprocesseings and run again
	for i in range(4):
		if pattern_count >= p:
			break
		else:
			if i == 0:
				preprocessed_image_1 = preprocess_1(gray)
				bb1 = ocr(preprocessed_image_1,cropped_image,des_fields)
				bb = bb+bb1
				pattern_count,bb = patterns_from_bb(bb)
			if i == 1:
				preprocessed_image_2 = preprocess_2(gray)
				bb2 = ocr(preprocessed_image_2,cropped_image,des_fields)
				bb = bb+bb2
				pattern_count,bb = patterns_from_bb(bb)
			if i == 2:
				preprocessed_image_3 = preprocess_3(gray)
				bb3 = ocr(preprocessed_image_3,cropped_image,des_fields)
				bb = bb+bb3
				pattern_count,bb = patterns_from_bb(bb)
			if i == 3:
				preprocessed_image_4 = preprocess_4(gray)
				bb4 = ocr(preprocessed_image_4,cropped_image,des_fields)
				bb = bb+bb4
				pattern_count,bb = patterns_from_bb(bb)

	return bb
