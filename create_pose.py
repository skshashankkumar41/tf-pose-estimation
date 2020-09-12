import argparse
import logging
import time
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
fps_time = 0

parser = argparse.ArgumentParser(description='pose-creator')
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
parser.add_argument('--path', type=str, default="",
                    help='for tensorrt process.')

args = parser.parse_args()

#args= {'resize':'432x368', 'model':'mobilenet_thin','camera':0,'resize_out_ratio':4.0}
w, h = model_wh(args.resize)

if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
else:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

cam = cv2.VideoCapture(args.camera)
ret_val, image = cam.read()
print("IMAGESHAPE:::",image.shape)
vid_writer = cv2.VideoWriter(args.path+'output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (image.shape[1],image.shape[0]),True)
while True:
    ret_val, image = cam.read()
    
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    image[image != 0] = 0
    #print("ZEROS::",image.shape)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)
    #print("SHAPE::",image.shape)

    # cv2.putText(image,
    #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
    #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 255, 0), 2)
    cv2.imshow('tf-pose-estimation result', image)
    fps_time = time.time()
    vid_writer.write(image)
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break

cv2.destroyAllWindows()
vid_writer.release()

