######################################################################
######################################################################
# Object Tracking & Object Counting by Line
# bounding box, label box 좌표 수정

# TEST #
'''
making csv
making function
'''
######################################################################
######################################################################

from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import cv2
from sort import *
import csv

# load weights and set defaults
# YOLO-v3
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'

img_size = 416
conf_thres = 0.8
nms_thres = 0.4

counter = 0
memory = {}

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)

'''Cuda Running'''
model.cuda()

model.eval()

classes = utils.load_classes(class_path)
print(classes)

'''Cuda Running'''
Tensor = torch.cuda.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

# videopath = 'C:/Users/md459/PycharmProjects/choiwb/BigData_Team_AI_Contest/CycleGAN/blackbox_day.mp4'

# videopath = 'park3.mp4'
videopath = 'highway.mp4'

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128),
          (128, 128, 0), (0, 128, 128)]

vid = cv2.VideoCapture(videopath)
# vid = cv2.VideoCapture(0)

# my smart phone real time tracking
# vid = cv2.VideoCapture("https://192.168.43.1:8080/video")
# vid = cv2.VideoCapture('http://192.168.40.122:8080/video')

mot_tracker = Sort()

cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800, 600))

# saving video
ret, frame = vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print("Video size: (%d, %d)" % (vw, vh))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

def make_line(shp1_x, shp1_y, shp2_x, shp2_y):
    return (int(shp1_x), int(shp1_y)), (int(shp2_x), int(shp2_y))

# frame별 세로 줄
# line_start, line_end = make_line(vw/2, 0, vw/2, vh)

# frame별 가로 줄
line_start, line_end = make_line(0, vh * 0.8, vw, vh * 0.8)

# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect_point(counter):
    # object별 기준 선 통과 좌표
    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))

    print('----------------------------------------------------')
    print('frame No: %d\n' % (frames))
    print('object center point : ', p0, p1)
    print('----------------------------------------------------')

    if intersect(p0, p1, line_start, line_end):
        counter += 1

    return p0, p1, str(counter)

def intersect_count():
    # detect line (green)
    cv2.line(frame, line_start, line_end, (0, 0xFF, 0), 3)

    # draw counter (frame별 기준 선 통과 한 객체 count
    cv2.putText(frame, str(counter), (int(vw * 0.1), int(vh * 0.2)), cv2.FONT_HERSHEY_DUPLEX, 5.0,
                (0, 255, 255), 3)

def save_csv():
    # making real time data frame
    data = {'realtime': [realtime], 'frames': [frames],'detecting object': [cls],
           'p0 counter point': [p0], 'p1 counter point': [p1], 'counter': [counter]}

    df = pd.DataFrame(data, columns=['realtime', 'frames', 'detecting object',
                                     'p0 counter point', 'p1 counter point', 'counter'])

    test_df = df.append(df, ignore_index = True)
    test_df.to_csv('test_df.csv', sep=',', encoding='utf-8')

    return test_df

frames = 0
frame_count = 0

det_start = time.time()

while (True):
    ret, frame = vid.read()
    if not ret:
        break

    frames += 1

    # frame 관련 2가지 중복 코드 (아래 코드와 중복 필요) BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    # frame 관련 2가지 중복 코드 (아래 코드와 중복 필요) RGB -> BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    boxes = []
    confidences = []
    classIDs = []

    if detections is not None:

        tracked_objects = mot_tracker.update(detections.cpu())
        # tracked_objects = mot_tracker.update(detections.cuda())

        box = detections[0:4]
        # box = detections[0] * [np.array(vw), detections[1] * np.array(vh), detections[2] * np.array(vw),
        # detections[3] * np.array(vh)]

        # box = detections[0:4] * Tensor([vw, vh, vw, vh])

        # unique_labels = detections[:, -1].cpu().unique()
        unique_labels = detections[:, -1].cuda().unique()

        n_cls_preds = len(unique_labels)

        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            realtime = datetime.datetime.now()

            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]

            cls = classes[int(cls_pred)]

            # all bounding box per 1 frame
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
            # object label box
            # cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 80, y1), color, -1)
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 30, y1), color, -1)

            # object label name
            # cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #            (255, 255, 255), 3)
            cv2.putText(frame, cls, (x1 + 15, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 3)

            # Real Time per frame
            cv2.putText(frame, str(realtime), (int(vw * 0.5), int(vh * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 3)

            print('----------------------------------------------------')
            print('frame No : %d\n' % (frames))
            print(realtime, cls + "-" + str(int(obj_id)))
            print('----------------------------------------------------')
            print('bounding box location :%s\n' % (tracked_objects[0:6]))
            print('----------------------------------------------------')

            ####################################################################################
            boxes.append([x1, y1, box_w, box_h])
            indexIDs.append(int(obj_id))
            memory[indexIDs[-1]] = boxes[-1]
            ####################################################################################

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))

                    ####################
                    intersect_point(counter)
                    intersect_count()
                    ####################

                i += 1

    # saving from video to image per frame
    # cv2.imwrite('output/frame%d.jpg' % (frame_count), frame)
    # frame_count += 1

    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

det_end = time.time()
det_result = det_end - det_start
print('영상 추적 걸린 시간 : %.4f (초)' % (det_result))

cv2.destroyAllWindows()
outvideo.release()