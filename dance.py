import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
fps_time = 0
args= {'resize':'432x368', 'model':'mobilenet_thin','camera':0,'resize_out_ratio':4.0}
w, h = model_wh(args['resize'])

if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args['model']), target_size=(w, h))
else:
    e = TfPoseEstimator(get_graph_path(args['model']), target_size=(432, 368))

cam = cv2.VideoCapture(args['camera'])
ret_val, image = cam.read()
vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (image.shape[1],image.shape[0]))
while True:
    ret_val, image = cam.read()

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args['resize_out_ratio'])
    image = np.zeros(image.shape)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow('tf-pose-estimation result', image)
    fps_time = time.time()
    vid_writer.write(image)
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break

cv2.destroyAllWindows()
vid_writer.release()

