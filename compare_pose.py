import argparse
import logging
import time
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
fps_time = 0

args= {'resize':'432x368', 'model':'mobilenet_thin','camera':0,'resize_out_ratio':4.0,'vid':"output.avi"}
w, h = model_wh(args['resize'])

if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args['model']), target_size=(w, h))
else:
    e = TfPoseEstimator(get_graph_path(args['model']), target_size=(432, 368))

cam = cv2.VideoCapture(args['camera'])
ret_val, image = cam.read()
cam_store = cv2.VideoCapture(args['vid'])
ret_val_store, image_store = cam_store.read()

while True:
    ret_val, image = cam.read()
    ret_val_store, image_store = cam_store.read()

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args['resize_out_ratio'])
    #print("ZEROS::",image.shape)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)

    final = cv2.bitwise_or(image, image_store)
    #print(np.nonzero(image_store).shape)
    # cv2.putText(image,
    #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
    #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 255, 0), 2)
    cv2.imshow('tf-pose-estimation result', final)
    fps_time = time.time()
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break

cv2.destroyAllWindows()