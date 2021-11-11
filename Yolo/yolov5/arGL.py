import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread

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

conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
imgdim = 0






texture_id = 0
thread_quit = 0
X_AXIS = 0.0
Y_AXIS = 0.0
Z_AXIS = 0.0
DIRECTION = 1

'''
cap = cv2.VideoCapture(0)
new_frame = cap.read()[1]
'''

source = '0'
imgsz = 512
stride = 64
dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
dataiter = iter(dataset)
path, img, im0s, vid_cap, s = next(dataset)
new_frame = im0s[0]

'''
def init():
    video_thread = Thread(target=update, args=())
    video_thread.start()
'''

def init_gl(width, height):
    global texture_id
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)


def update():
    global new_frame
    while(True):
        #new_frame = cap.read()[1]
        path, img, im0s, vid_cap, s = next(dataset)
        new_frame = img

        if thread_quit == 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def draw_gl_scene():
    global cap
    global new_frame
    global X_AXIS, Y_AXIS, Z_AXIS
    global DIRECTION
    global texture_id
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    frame = new_frame

    global path, img, im0s
    path, img, im0s, vid_cap, s = next(dataset)
    new_frame = im0s[0]

    cvyolo(img)


    # convert image to OpenGL texture format
    tx_image = cv2.flip(frame, 0)
    tx_image = Image.fromarray(tx_image)
    ix = tx_image.size[0]
    iy = tx_image.size[1]
    tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)
    # create texture
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPushMatrix()
    glTranslatef(0.0, 0.0, -6.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(4.0, 3.0, 0.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-4.0, 3.0, 0.0)
    glEnd()
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0.0, 0.0, -6.0)
    glRotatef(X_AXIS, 1.0, 0.0, 0.0)
    glRotatef(Y_AXIS, 0.0, 1.0, 0.0)
    glRotatef(Z_AXIS, 0.0, 0.0, 1.0)

    glBegin(GL_QUADS)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)

    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)
    
    glColor3f(1.0, 1.0, 1.0)
    glEnd()
    glPopMatrix()
    X_AXIS = X_AXIS - 0.30
    Z_AXIS = Z_AXIS - 0.30

    glutSwapBuffers()


def key_pressed(key, x, y):
    global thread_quit
    if key == chr(27) or key == "q":
        thread_quit = 1
        sys.exit()


def cvyolo(img):
    global dataset
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
                line = (cls, *xywh, conf)  # label format

                # line is a tuple containing the results
                print('has the following ', line)                 
                resultboxes.append(line)



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

    # Run inference
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    global imgdim 
    imgdim = imgsz
 


def run():

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(200, 200)
    window = glutCreateWindow('My and Cube')
    glutDisplayFunc(draw_gl_scene)
    glutIdleFunc(draw_gl_scene)
    glutKeyboardFunc(key_pressed)
    init_gl(640, 480)
    glutMainLoop()


if __name__ == "__main__":
    #init()
    initcv(imgsz=[500, 500], weights=['/home/hpc/FYPdental/Yolo/yolov5/runs/train/exp4/weights/last.pt'])
    run()
