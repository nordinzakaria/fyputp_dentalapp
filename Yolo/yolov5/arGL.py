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

frameindex = 0

toolcolor = {
        "ToolA": ([80, 60, 125], [95, 85, 140])
        }

toolspots = []

class BBox:
    ID = 0
    def __init__(self, x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x + w
        self.y2 = y + h
        self.consumed = False
        self.id = BBox.ID
        BBox.ID += 1

    def consume(self, ctr2):
        if self.x < ctr2.x and self.x2 < ctr2.x2:   # entirely left
            self.w = ctr2.x2 - self.x
        elif ctr2.x < self.x and ctr2.x2 < self.x2: # entirely right
            self.w = self.x2 - ctr2.x
            self.setX(ctr2.x)
        elif ctr2.x < self.x and ctr2.x2 > self.x2: # engulfed by ctr2
            self.w = ctr2.w
            self.setX(ctr2.x)

        if self.y < ctr2.y and self.y2 < ctr2.y2:   # entirely lower
            self.h = ctr2.y2 - self.y
        elif ctr2.y < self.y and ctr2.y2 < self.y2: # entirely higher
            self.h = self.y2 - ctr2.y
            self.setY(ctr2.y)
        elif ctr2.y < self.y and ctr2.y2 > self.y2: # engulfed by ctr2
            self.h = ctr2.h
            self.setY(ctr2.y)


        ctr2.consumed = True

    def setX(self, x):
        self.x = x
        self.x2 = x + self.w

    def setY(self, y):
        self.y = y
        self.y2 = y + self.h

    def print(self):
        print("Box ", self.id, " has [", self.x, ",", self.y, ",", self.w, ",", self.h, "]")


    def isNear(self, ctr2):
        center = np.array((0,0))
        center[0] = (self.x + self.x2) / 2
        center[1] = (self.y + self.y2) / 2

        center2 = np.array((0,0))
        center2[0] = (ctr2.x + ctr2.x2) / 2
        center2[1] = (ctr2.y + ctr2.y2) / 2

        dist = np.linalg.norm(center - center2)
        if dist < 100:
            return True

        return False





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
    detectTool(frame)
    '''
    global frameindex
    cv2.imwrite("lihat-"+str(frameindex)+".jpg", frame)
    frameindex += 1
    '''

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
    glScalef(0.3, 0.3, 0.3)

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

def detectTool(image):
    global toolspots
    toolspots.clear()

    for tool in toolcolor.keys():
        (lower, upper) = toolcolor[tool]
            # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)

        output = cv2.medianBlur(output, 5)
        kernel = np.ones((5,5),np.uint8)
        cv2.dilate(output,kernel,iterations = 1)


        gray=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # collect contours into a list
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue
            result = BBox(x,y,w,h)
            result.print()
            toolspots.append(result)

    while True: # repeat for as long as some merge is possible
        hasMerge = False

        for ctr in toolspots:   # N^2 loop
            if ctr.consumed == True:
                continue

            for ctr2 in toolspots:
                if ctr.id == ctr2.id:
                    continue

                if ctr.consumed == True:
                    continue

                near = ctr.isNear(ctr2)
                if near == True:
                    ctr.consume(ctr2)
                    hasMerge = True

        if hasMerge == False:
            break

        newlist = []    
        for ctr in toolspots:
            if ctr.consumed == True:
                continue
            newlist.append(ctr)
        toolspots = newlist

    print("After merging.. ")
    for ctr in toolspots:
        ctr.print()





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
