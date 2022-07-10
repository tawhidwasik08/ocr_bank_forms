# OCR Bank Form Fields

This repository seeks to find certain data fields containing handwritings in certain bank forms.

## Environment Setup
* Use `conda env create -f env.yml` to create new virtual environment.

* Please be noted the environment was created in windows.


## Image Directory Setup
* Make sure the images are in the __root folder__ in the  following way:
   	* __folder name/format\_<format number>/variant\_<variant number>/__
   * Example : __samples/format\_1/variant\_1/22.png__


## Run 
* Run _ocr_fields_extract.py --dir image_folder_directory_ from command line.
* Example (conda prompt): `python ocr_fields_extract.py -dir samples/format_1/variant_1/`

## Output

* Output cropped images will be saved in the extract_image_field/respective format/respective variant/image file name/ in the root directory.


## Privacy 
_Due to privacy issues, sample images and outputs are not added to repository._
