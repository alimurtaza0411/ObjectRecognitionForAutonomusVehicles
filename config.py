import os

ORIG_BASE_PATH = "dataset"
ORIG_IMAGES    = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS    = os.path.sep.join([ORIG_BASE_PATH, "annotations.csv"])

BASE_PATH      = "cropped_dataset"
POSITIVE_PATH  = os.path.sep.join([BASE_PATH, "vehicle"])
NEGATIVE_PATH  = os.path.sep.join([BASE_PATH, "no_vehicle"])

MODEL_PATH     = "vehicle_detector.h5"
ENCODER_PATH   = "label_encoder.pickle"

MIN_PROBA      = 0.95

MAX_PROPOSALS  = 2000
IOU_THRESHOLD  = 0.05
COUNT_OF_DETECT= 5
INPUT_DIMS     = (200, 200)

DATASET_IMAGE_SIZE    = (512,512)
MIN_DIM_TO_CONSIDER   = 50
NEGATIVE_SAMPLE_COUNT = 2
