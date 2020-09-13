import argparse
import logging
import time
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
from scipy.spatial import distance
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from sklearn.preprocessing import Normalizer
fps_time = 0

parser = argparse.ArgumentParser(description='pose-comparator')
parser.add_argument('--video', type=str, default='output.avi')
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
parser.add_argument('--tensorrt', type=str, default="False",
                    help='for tensorrt process.')
args = parser.parse_args()

#args= {'resize':'432x368', 'model':'mobilenet_thin','camera':0,'resize_out_ratio':4.0,'video':"output.avi"}
w, h = model_wh(args.resize)

if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
else:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

out = False
cam = cv2.VideoCapture(args.camera)
ret_val, image = cam.read()
cam_store = cv2.VideoCapture(args.video)
ret_val_store, image_store = cam_store.read()

while cam_store.isOpened():
    dist = True 
    ret_val_store, image_store = cam_store.read()
    print("-------Next Pose Applied-------")
    while dist:
        ret_val, image = cam.read()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        #print("ZEROS::",image.shape)

        image, blackImage = TfPoseEstimator.draw_humans(image, humans, imgcopy=True,black_image = True)
        # import matplotlib.pyplot as pltqqqqqqqqq
        # plt.imsave('test2.jpg', image, cmap='Greys')
        
        final = cv2.bitwise_or(image, image_store)
        cv2.imshow('tf-pose-estimation result', final)
        #print(np.nonzero(image_store).shape)
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)q
        #cv2.imshow('tf-pose-estimation result', final)
        norm = Normalizer()
        Aflat = norm.fit_transform(blackImage.flatten(order='C'))
        Bflat = norm.transform(image_store.flatten(order='C'))

        dist = distance.cosine(Aflat, Bflat)
        print("Similarity:",dist)
        if dist > 0.90:
            dist = False
        
        fps_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            out = True
            break
            
    if cv2.waitKey(5) & out: 
            break

cv2.destroyAllWindows()