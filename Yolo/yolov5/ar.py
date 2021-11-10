import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import load_classifier, select_device, time_sync




from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
imgdim = 0


w,h= 500,500
def square():

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(100, 100)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(300, 100.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(300, 300)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(100, 300)
    glEnd()

def iterate():
    glViewport(0, 0, 500, 500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluOrtho2D(0.0, 500, 0.0, 500)
    glMatrixMode (GL_MODELVIEW)
    glLoadIdentity()

def showScreen():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    iterate()

    glColor3f(1.0, 0.0, 3.0)
    square()
    glutSwapBuffers()


def cvyolo():
    global dataset
    path, img, im0s, vid_cap, s = next(dataset)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32


    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference

    global model
    pred = model(img, augment=False, visualize=False)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        resultboxes = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                # line is a tuple containing the results
                print('has the following ', line)                 
                resultboxes.append(line)

    global imgdim
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
      imgdim,imgdim,
      0,
      GL_RGB, 
      GL_FLOAT, 
      img)



def initcv(imgsz, weights):
    source = '0'

    # Initialize
    global device
    device=''
    device = select_device(device)

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    pt = True

    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    global model
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    global dataset, dataiter
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
    dataiter = iter(dataset)

    # Run inference
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    global imgdim 
    imgdim = imgsz
 

def initgl():
    glEnable(GL_TEXTURE_2D)
    #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    #glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    #this one is necessary with texture2d for some reason
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)


def display():
    initcv(imgsz=[500, 500], weights=['/home/hpc/FYPdental/Yolo/yolov5/runs/train/exp4/weights/last.pt'])
    glutInit()
    initgl()

    glutInitDisplayMode(GLUT_RGBA)
    glutInitWindowSize(500, 500)
    glutInitWindowPosition(0, 0)
    wind = glutCreateWindow("OpenGL Coding Practice")
    glutDisplayFunc(showScreen)
    glutIdleFunc(cvyolo)
    glutMainLoop()

display()
