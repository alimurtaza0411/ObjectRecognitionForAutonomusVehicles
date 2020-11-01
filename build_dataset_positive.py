# import the necessary packages
import os
import cv2
import iou
import config
import pandas as pd
from imutils import paths
from iou import compute_iou
from collections import defaultdict

# if the output directory does not exist yet, create it
if not os.path.exists(config.POSITIVE_PATH):
	os.makedirs(config.POSITIVE_PATH)


# grab all image paths in the input images directory
imagePaths = list(paths.list_images(config.ORIG_IMAGES))

#read the csv and convert to dictionary
annotPath = config.ORIG_ANNOTS
annotsDf  = pd.read_csv(annotPath).values
annots    = defaultdict(list)

for i in range(annotsDf.shape[0]):
	fileName = annotsDf[i][0]
	annots[fileName].append(annotsDf[i][1:])

# initialize the total number of positive and negative images we have saved to disk so far
totalPositive = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):

	# show a progress report
	print("[INFO] processing image {}/{}...".format(i + 1,len(imagePaths)))

	# extract the filename from the file path and use it to get its bounding boxes from the annotations file
	filename = imagePath.split(os.path.sep)[-1]

	w     = config.DATASET_IMAGE_SIZE[0]
	h     = config.DATASET_IMAGE_SIZE[1]
	image = cv2.imread(config.ORIG_IMAGES+'/'+filename)

	# loop over all boxes in the image
	for box in annots[filename]:
		
		if box[2]=="car" or box[2]=="truck":

			# extract the bounding box coordinates
			xMin = box[3]
			yMin = box[4]
			xMax = box[5]
			yMax = box[6]
		
			# truncate any bounding box coordinates that may fall outside the boundaries of the image
			xMin = max(0, xMin)
			yMin = max(0, yMin)
			xMax = min(w, xMax)
			yMax = min(h, yMax)
			
			if xMax-xMin>=config.MIN_DIM_TO_CONSIDER and yMax-yMin>=config.MIN_DIM_TO_CONSIDER:

				roi = image[yMin:yMax, xMin:xMax]
				roi = cv2.resize(roi,config.INPUT_DIMS,interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(config.POSITIVE_PATH+'/'+str(totalPositive)+'.jpg', roi)
				totalPositive+=1