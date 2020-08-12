# OCR Bank Form Fields

This repository seeks to find certain data fields containing handwritings in NBL bank forms.

## Environment Setup
* Use `conda env create -f env.yml` to create new virtual environment.

* Please be noted the environment was created in windows.


## Image Directory Setup
* Make sure the images are in the __root folder__ in the  following way:
   	* __folder name/format\_<format number>/variant\_<variant number>/__
   * Example : __samples/format\_1/variant\_1/22.png__


## Run 
* Run _ocr_fields.py -dir image_folder_directory_ from command line.
* Example (conda prompt): `python ocr_fields.py -dir samples/format_1/variant_1/`

## Output
* Fields are shown in blue rectangle regions in output images.
* OCR'ed texts of the desired fields are in red rectangles.
* Output images will be saved in the output_image/respective format and variant folder in the root directory.


## Privacy 
_Due to privacy issues, sample images and outputs are not added to repository._
