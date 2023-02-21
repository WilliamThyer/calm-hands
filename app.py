## tkinter app that opens a window and displays your webcam feed

import tkinter as tk
import cv2
import PIL.Image as Image, PIL.ImageTk as ImageTk
import time
import numpy as np
from dummy_model import DummyModel
from pygame import mixer

from fastai.vision.all import *
from fastai.vision.utils import *

# Weird path stuff for fastai
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def label_func(self,name):
    # get label from file name (needed for fastai)
    return name.parent.name

class App:
    
    def __init__(self, dummy=False):
        
        self.cap = None
        self.window = None
        self.video_frame = None
        self.pred_probs = []
        self.model = None
        self.run_preds = True
        self.show_webcam = True
        self.dummy = dummy
        if dummy:
            global load_learner
        self.load_sound()
    
    def start(self):

        self.create_window()
        self.create_video()
        self.create_show_vid_button()
        self.create_run_preds_button()
        self.create_text()
        self.create_chart()
        self.load_model()
        self.start_webcam()
        self.show_frame()
        self.predict()

    def create_window(self):
        self.window = tk.Tk()
        self.window.title("Calm Hands")
        self.window.resizable(100,100)
        self.window.configure(background='black')

    # SOUND STUFF
    def load_sound(self):
        mixer.init()
        self.sound = mixer.Sound("alert.wav")
    
    def play_sound(self, pred, pred_prob):
        if (pred_prob > 0.8) and (pred[0] == 'bad'):
            self.sound.play()

    # TEXT STUFF
    def create_text(self):
        self.text = tk.Label(self.window, text="", fg="white", bg="black")
        self.text.grid(row=0, column=0, padx=10, pady=2)
        
    def update_text(self,text):
        self.text.configure(text=text)

    # MODEL STUFF
    def load_model(self,model_path = 'edgenext_model.pkl'):
        # load pretrained model
        print('Loading model...')
        if self.dummy:
            self.model = DummyModel()
        else:
            self.model = load_learner(model_path,cpu=True)
        print('Model loaded!')

    def create_output(self,pred):
        # create output string
        conf = round(float(pred[2][pred[1]])*100,2) 
        output = f'The model is {conf}% confident that you are {pred[0]}' 
        return output
    
    def do_prediction(self,frame):
        # do prediction on frame
        with self.model.no_bar():
            return self.model.predict(frame)

    def predict(self):
        # predict on webcam feed
        if (self.model) and (self.run_preds):
            pred = self.do_prediction(self.raw_frame)
            pred_prob = float(pred[2][pred[1]])
            self.pred_probs.append(pred_prob)
            pred_str = self.create_output(pred)
            self.play_sound(pred, pred_prob)
        else:
            pred_str = "Paused"
        
        self.update_text(pred_str)

        self.window.after(500, self.predict)

    # VIDEO STUFF

    def create_video(self):
        self.video_frame = tk.Label(self.window, bg="black")
        self.video_frame.grid(row=1, column=0, padx=10, pady=2)

    def show_frame(self):

        _, self.raw_frame = self.cap.read()
        if self.show_webcam:
            frame = cv2.flip(self.raw_frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            frame = Image.fromarray(frame) # convert to PIL image
            frame = ImageTk.PhotoImage(frame) # convert to PhotoImage
            self.video_frame.configure(image=frame) # display the image
            self.video_frame._image_cache = frame # avoid garbage collection
        
        self.window.after(15, self.show_frame) # 15 ms delay
    
    def start_webcam(self):
        # start webcam feed
        if self.cap is None:
            print('Starting webcam...')
            self.cap = cv2.VideoCapture(0) # 0 is the default camera
    
    def stop_webcam(self):
        # stop webcam feed
        self.cap.release()
        self.video_frame.configure(image='')
        self.video_frame._image_cache = None
        self.cap = None
    
    # CHART STUFF
    def create_chart(self):
        self.chart = tk.Canvas(self.window, width=680, height=480)
        self.chart.grid(row=1, column=1, padx=10, pady=2)
    
    # def create_plot(self):
    #     # the figure that will contain the plot
    #     fig = Figure(figsize = (5, 5),
    #                 dpi = 100)
    
    #     # adding the subplot
    #     plot1 = fig.add_subplot(111)

    # def plot_preds():
    #     # plotting the graph
    #     plot1.plot(y)
    
    #     # creating the Tkinter canvas
    #     # containing the Matplotlib figure

    #     canvas = FigureCanvasTkAgg(fig,
    #                             master = window)  
    #     canvas.draw()
    
    #     # placing the canvas on the Tkinter window
    #     canvas.get_tk_widget().pack()
    
    #     # creating the Matplotlib toolbar
    #     toolbar = NavigationToolbar2Tk(canvas,
    #                                 window)
    #     toolbar.update()
    
    #     # placing the toolbar on the Tkinter window
    #     canvas.get_tk_widget().pack()

    # BUTTONS
    def create_show_vid_button(self):
        self.show_vid_button = tk.Button(self.window, text="Show/Hide Video", command=self.switch_show_webcam)
        self.show_vid_button.grid(row=2, column=0, padx=10, pady=2)
    
    def create_run_preds_button(self):
        self.preds_button = tk.Button(self.window, text="Play/Pause Predictions", command=self.switch_run_preds)
        self.preds_button.grid(row=2, column=1, padx=10, pady=2)
    
    def switch_run_preds(self):

        if self.run_preds is False:
            self.run_preds = True
        else:
            self.run_preds = False

    def switch_show_webcam(self):

        if self.show_webcam is False:
            self.show_webcam = True
        else:
            self.show_webcam = False
            self.video_frame.configure(image='')
            self.video_frame._image_cache = None                

    # DUNDER METHODS
    def __del__(self):
        self.stop_webcam()
        self.window.destroy()
    
    def __exit__(self):
        self.stop_webcam()
        self.window.destroy()

    def run(self):
        self.start()
        self.window.mainloop()

if __name__ == '__main__':
    app = App(dummy=False)
    app.run() 