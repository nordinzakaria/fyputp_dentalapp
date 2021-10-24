#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#GUI SIMPLE - WORKING

# Import the tkinter module
import tkinter as tk
import webbrowser
import cv2
import PIL.Image, PIL.ImageTk

HEIGHT = 700
WIDTH = 800


import tkinter as tk
from tkinter import ttk


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
 
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)


# New: class for learning
class LearnWindow(tk.Toplevel):
    def __init__(self, parent=None,  video_source=0):
        super().__init__(parent)

        # open video source
        self.vid = MyVideoCapture(video_source)

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
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        
        self.after(self.delay, self.update)

new = 1
url = "https://github.com/nuruladilah/DentalCleaning-New-"

def openbrowser():
    webbrowser.open(url,new=new)


        # New: create learning window
def open_lwindow():
        lwindow = LearnWindow(video_source='example.avi')
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

