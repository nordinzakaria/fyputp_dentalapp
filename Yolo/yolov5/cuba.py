import torch
import torch.nn as nn
import torch.optim as optim

# Model

PATH = '/home/hpc/FYPdental/Yolo/yolov5/runs/train/exp4/weights/last.pt'
model = Net()
model.load_state_dict(torch.load(PATH))


print('model = ', model)

img = '/home/hpc/FYPdental/Yolo/DentalCleaning-New-/Test/images/Frame_1179.JPG'  # or file, Path, PIL, OpenCV, numpy, list

model.eval()

