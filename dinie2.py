#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#GUI SIMPLE - WORKING

# Import the tkinter module
import tkinter as tk
import webbrowser
import cv2
import PIL.Image, PIL.ImageTk
import pandas as pd
import os
import glob

HEIGHT = 700
WIDTH = 800


import tkinter as tk
from tkinter import ttk


class MyVideoCapture:
    def __init__(self, frame_source='./Dental_frame/images/', label_source='./Dental_frame/annotations/', prefix='dentalframe', suffix='.jpg', startframe=0):
        self.framefolder = frame_source
        self.labelfolder = label_source
        self.prefix = prefix
        self.suffix = suffix
        self.startframe = startframe

        self.numframe = len(glob.glob(self.framefolder + self.prefix + '*' + self.suffix)) 
        print ('num frames = ', self.numframe)
        self.currframe = self.startframe

        frame0 = cv2.imread(self.framefolder + self.prefix + str(self.currframe) + self.suffix)
        h, w, c = frame0.shape
        self.width = w
        self.height = h
        print('created window of size ', w, ' x ', h)


    # Release the video source when the object is destroyed
    def __del__(self):
        pass

    def get_frame(self):
        if self.currframe >= self.numframe:
            return (False, False, None, None)

        path = self.framefolder + self.prefix + str(self.currframe) + self.suffix
        csvpath = self.labelfolder + self.prefix + str(self.currframe) + ".txt"
        frame = None
        df = None
        ret = False
        retcsv = False
        if os.path.isfile(path):
            frame = cv2.imread(path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret = True
            if os.path.isfile(csvpath):
                df = pd.read_csv (csvpath, header=None, sep=' ')
                retcsv = True

        self.currframe += 1
        return (ret, retcsv, frame, df)

# list of bounding boxes for current frame
bboxes = []

# New: class for learning
class VideoWindow(tk.Toplevel):
    def __init__(self, parent=None):
        super().__init__(parent)

        # open video source
        self.vid = MyVideoCapture()

        #self.geometry('600x600')
        self.geometry(str(int(self.vid.width))+'x'+str(int(self.vid.width)))
        self.title('Learn')


        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()      

    def update(self):
        # Get a frame from the video source
        ret, retcsv, frame, df = self.vid.get_frame()

        if ret == True:
            # draw/write whatever on frame


            if retcsv == True:

                global bboxes
                bboxes.clear()

                for ind in df.index:
                    classlabel = int(df.at[ind, 0])
                    xmin = int(df.at[ind, 1] * self.vid.width)
                    ymin = int(df.at[ind, 2] * self.vid.height)
                    xmax = int(df.at[ind, 3] * self.vid.width) + xmin
                    ymax = int(df.at[ind, 4] * self.vid.height) + ymin
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)

                    bboxes.append((classlabel, xmin, ymin, xmax, ymax))


            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        
        self.after(self.delay, self.update)

new = 1
url = "https://github.com/nuruladilah/DentalCleaning-New-"

def openbrowser():
    webbrowser.open(url,new=new)


def mouseclicked(event):
    print ("clicked at", event.x,',', event.y)
    global bboxes
    for box in bboxes:
        if event.x < box[1] or event.x > box[3]:
            continue
        if event.y < box[2] or event.y > box[4]:
            continue

        print ('Clicked on a box with label ', box[0])

        # New: create learning window
def open_lwindow():
        lwindow = VideoWindow()
        lwindow.bind("<Button-1>", mouseclicked)
        lwindow.grab_set()
        



root = tk.Tk()

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

                        # New/ Note: tukar balik to your own background image
background_image = tk.PhotoImage(file=r'/home/hpc/Pictures/PNG.png')
background_label =tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

frame = tk.Frame(root, bg='#524945')
frame.place(relx= 0.1, rely=0.1, relwidth=0.8, relheight=0.8)

button = tk.Button(frame, text = "LEARN MODE", font=20, 
        #command=openbrowser,)
        command=open_lwindow)  #New: call open_window function

button.place(relx= 0.1, rely= 0.4, relwidth=0.3, relheight=0.3)

button1 = tk.Button(frame, text = "QUIZ MODE", font=20)
button1.place(relx= 0.6, rely= 0.4, relwidth=0.3, relheight=0.3)

label = tk.Label(frame, text="WELCOME TO DENTAL GUI TESTING", font=100)
label.pack()

label1 = tk.Label(frame, text="LET'S GET STARTED!", font=50)
label1.place(rely= 0.2, relx= 0.27)

#https://www.geeksforgeeks.org/how-to-set-font-for-text-in-tkinter/
# Creating a tuple containing 
# the specifications of the font.
Font_tuple = ("Comic Sans MS", 20, "bold")
  
# Parsed the specifications to the
# Text widget using .configure( ) method.
label.configure(font = Font_tuple)
label1.configure(font = Font_tuple)
button.configure(font = Font_tuple)
button1.configure(font = Font_tuple)

root.mainloop()

