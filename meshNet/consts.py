import os

TOP_DIR = os.path.dirname(__file__) + os.sep

OUTPUT_DIR = os.path.join(TOP_DIR, 'sessions_outputs')

IMAGE_RANGE = (0, 255)

# Dimensions are according to TensorFlow convention
BATCH_DIMS_SAMPLES = 0
BATCH_DIMS_ROWS = 1
BATCH_DIMS_COLS = 2
BATCH_DIMS_CHANNELS = 3

IMAGE_DIMS_ROWS = 0
IMAGE_DIMS_COLS = 1
IMAGE_DIMS_CHANNELS = 2
