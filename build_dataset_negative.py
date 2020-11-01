# import the necessary packages
import os
import cv2
import iou
import config
import random
import pandas as pd
from imutils import paths
from iou import compute_iou
from collections import defaultdict

# if the output directory does not exist yet, create it
if not os.path.exists(config.NEGATIVE_PATH):
	os.makedirs(config.NEGATIVE_PATH)


# grab all image paths in the input images directory
imagePaths = list(paths.list_images(config.ORIG_IMAGES))

# initialize the total number of negative images we have saved to disk so far
totalNegatives = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):

	# show a progress report
	print("[INFO] processing image {}/{}...".format(i + 1,len(imagePaths)))

	# extract the filename from the file path and use it to get its bounding boxes from the annotations file
	filename = imagePath.split(os.path.sep)[-1]

	image = cv2.imread(config.ORIG_IMAGES+'/'+filename)

	for i in range(config.NEGATIVE_SAMPLE_COUNT):
		x = random.randint(0,config.DATASET_IMAGE_SIZE[0]-config.MIN_DIM_TO_CONSIDER*2)
		y = random.randint(0,config.DATASET_IMAGE_SIZE[1]-config.MIN_DIM_TO_CONSIDER*2)
		roi = image[y:y+config.MIN_DIM_TO_CONSIDER*2,x:x+config.MIN_DIM_TO_CONSIDER*2]
		roi = cv2.resize(roi,config.INPUT_DIMS,interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(config.NEGATIVE_PATH+'/'+str(totalNegatives)+'.jpg', roi)
		totalNegatives+=1