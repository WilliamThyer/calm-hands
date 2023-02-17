## tkinter app that opens a window and displays your webcam feed

import tkinter as tk
import cv2
import PIL.Image as Image, PIL.ImageTk as ImageTk
import time
import numpy as np
from dummy_model import DummyModel

# from fastai.vision.all import *
# from fastai.vision.utils import *

# Weird path stuff for fastai
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def label_func(self,name):
    # get label from file name (needed for fastai)
    return name.parent.name

class App:
    
    def __init__(self, dummy=False):
        # initialize variables
        self.cap = None
        self.window = None
        self.video_frame = None
        self.text = None
        self.button = None
        self.button2 = None
        self.pred_probs = []
        self.model = None
        self.dummy = dummy
        if dummy:
            global load_learner 
    
    def load_model(self,model_path = 'edgenext_model.pkl'):
        # load pretrained model
        print('Loading model...')
        if self.dummy:
            model = DummyModel()
        else:
            model = load_learner(model_path,cpu=True)
        print('Model loaded!')
        return model

    def create_output(self,pred):
        # create output string
        conf = round(float(pred[2][pred[1]])*100,2) 
        output = f'The model is {conf}% confident that you are {pred[0]}' 
        return output
    
    def do_prediction(self,frame):
        # do prediction on frame
        return self.model.predict(frame,no_bar=True)

    def start(self):
        # start the window 
        self.window = tk.Tk()
        self.window.title("Calm Hands")
        self.window.resizable(100,100)
        self.window.configure(background='black')
        self.video_frame = tk.Label(self.window)
        self.video_frame.pack()
        self.create_start_button()
        self.create_stop_button()
        self.add_text()
        self.create_chart()
        self.model = self.load_model()
    
    def add_text(self):
        self.text = tk.Label(self.window, text="", fg="black", bg="white")
        self.text.pack()
        
    def update_text(self,text):
        self.text.configure(text=text)
    
    def show_frame(self):
        _, raw_frame = self.cap.read()
        frame = cv2.flip(raw_frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        frame = Image.fromarray(frame) # convert to PIL image
        frame = ImageTk.PhotoImage(frame) # convert to PhotoImage
        self.video_frame.configure(image=frame) # display the image
        self.video_frame._image_cache = frame # avoid garbage collection
        self.window.after(15, self.show_frame) # 15 ms delay

        if self.model is not None:
            pred = self.do_prediction(raw_frame)
            pred_prob = float(pred[2][pred[1]])
            self.pred_probs.append(pred_prob)
            pred_str = self.create_output(pred)
        else:
            pred_str = "Loading model..."

        self.update_text(pred_str)
        
    def start_webcam(self):
        # start webcam feed
        if self.cap is None:
            self.cap = cv2.VideoCapture(0) # 0 is the default camera
            self.show_frame()
    
    def stop_webcam(self):
        # stop webcam feed
        self.cap.release()
        self.video_frame.configure(image='')
        self.video_frame._image_cache = None
        self.cap = None
        self.update_text("Stopped")
    
    def close_window(self):
        self.window.destroy()
    
    def run(self):
        self.start()
        self.start_webcam()
        self.window.mainloop()
        
    def create_chart(self):
        self.chart = tk.Canvas(self.window, width=200, height=100)
        self.chart.pack()
     
    def update_chart(self):
        # draw line chart with self.pred_probs
        self.chart.create_line(0,0,1,1, fill="red", width=2)
        self.chart.update()
    
    def create_start_button(self):
        self.button = tk.Button(self.window, text="Start", command=self.start_webcam)
        self.button.pack()
    
    def create_stop_button(self):
        self.button2 = tk.Button(self.window, text="Stop", command=self.stop_webcam)
        self.button2.pack()
    
    def create_button4(self):
        self.button4 = tk.Button(self.window, text="Run", command=self.run)
        self.button4.pack()
        
    def __del__(self):
        self.stop_webcam()
        self.close_window()
    
    def __exit__(self):
        self.stop_webcam()
        self.close_window()
    
    def __enter__(self):
        self.start()
        self.start_webcam()
        self.window.mainloop()
        return self

if __name__ == '__main__':
    app = App(dummy=True)
    app.run() 