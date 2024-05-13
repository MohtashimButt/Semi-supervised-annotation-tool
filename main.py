from cropFromBbox import process_json_files 
import os
import cv2
from masktojson import mask_to_json
from model import get_inference
from unLabelToLabel import labelResize
from masktojson import jasonify

json_folder = "Labelme_bbox"
labels_folder = "Labels"
jsons_folder = "JSONS"
unlabelled_images_folder = "Unlabelled_Images"

# crop the bbox (will save masks from `json_folder` to 'Unlabelled_Images')
process_json_files(json_folder)

# mask the imges (will save masks from `unlabelled_images_folder` to "Labels")
get_inference(unlabelled_images_folder)

# separate the resized images as labeled images
labelResize(unlabelled_images_folder)

# jsonify those masks (will make the JSONS out of masks in Labels folder)
jasonify(labels_folder)

print("NOW OPEN LABELME WITH IMAGES AS INPUT DIR AND LABELS AS OUTPUT DIR")