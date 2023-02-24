import tkinter as tk
import customtkinter as ctk
ctk.set_appearance_mode('dark')

import cv2
import PIL.Image as Image, PIL.ImageTk as ImageTk
import time
import numpy as np
from dummy_model import DummyModel
from pygame import mixer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

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
        self.pred_str = ''
        self.model = None
        self.run_preds = True
        self.show_webcam = True
        self.fig = None
        
        # Time stuff
        self.timer = 0
        self.hz = 2
        self.pred_wait = int(1000/self.hz)
        self.view_secs = 60
        self.len_view = int(self.view_secs*self.hz)
        
        self.dummy = dummy
        if dummy:
            global load_learner
        
        self.load_sound()
    
    def start(self):

        self.create_window()
        self.create_video()
        self.create_show_vid_button()
        self.create_run_preds_button()
        self.create_end_session_button()
        self.create_plot()
        self.load_model()
        self.start_webcam()
        self.show_frame()
        self.predict()
        self.update_plot()

    def create_window(self):
        self.window = ctk.CTk()
        self.window.title("Calm Hands")
        self.window.resizable(100,100)

    # SOUND STUFF
    def load_sound(self):
        mixer.init()
        self.sound = mixer.Sound("alert.wav")
    
    def play_sound(self, pred, pred_prob):
        if (pred_prob > 0.75) and (pred[0] == 'bad'):
            self.sound.play()

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
            self.pred_probs.append(float(pred[2][1])) # probability of good
            self.timer += 1/self.hz
            self.pred_str = self.create_output(pred)
            self.play_sound(pred, pred[2][pred[1]])
        else:
            pred_str = "Paused"
        
        self.window.after(self.pred_wait, self.predict)

    # VIDEO STUFF
    def create_video(self):
        self.video_frame = tk.Label(self.window, bg="black")
        self.video_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=2)

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
    
    # PLOT STUFF
    def format_preds_for_plot(self, preds):

        if len(preds) > self.len_view:
            preds = preds[-self.len_view:]
            times = np.arange(self.timer-self.view_secs,self.timer,1/self.hz)
            return preds, times
        if len(preds) == 0:
            preds = [0]
        times = np.arange(0,len(preds)/self.hz,1/self.hz)
        if len(times) > len(preds):
            times = times[:-1]
        return preds, times
    
    def make_plot(self, preds, times):

        if self.fig is None:
            self.fig, ax = plt.subplots(figsize=(6.4,4.8),dpi=100)
        else:
            ax = self.fig.axes[0]
            ax.clear()

        ax.plot(times,preds,color='black',zorder=2,linewidth=1.75)
        ax.plot(times,preds, color='white',zorder=1,linewidth=5)

        ax.spines[['right', 'top']].set_visible(False)
        ax.set(xlabel='Time (s)',ylabel='');
        ax.set_ylim(0,1.01)
        ax.set_yticks([])

        if self.timer > self.view_secs:
            ax.set_xlim(min(times),max(times))
        else: 
            ax.set_xlim(0,self.view_secs)
        start, end = ax.get_xlim()

        plt.text(-.02, 0.75, 'Good', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,rotation=90)
        plt.text(-.02, 0.125, 'Bad', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,rotation=90)

        # shade upper half of plot green
        ax.fill_between(np.arange(start,end+1), .5, 1.01, facecolor='green', alpha=0.2, zorder=-2)
        ax.fill_between(np.arange(start,end+1), 0, .5, facecolor='red', alpha=0.2, zorder=-2)
        ax.fill_between(np.arange(start,end+1), 0, .2, facecolor='red', alpha=0.2, zorder=-2)

        ax.set_title(self.pred_str,fontsize=16)

        plt.tight_layout()

    def create_plot(self):

        if self.run_preds:

            # plotting the graph
            preds, times = self.format_preds_for_plot(self.pred_probs)
            self.make_plot(preds, times)
                
            # creating the Tkinter canvas containing the Matplotlib figure
            self.canvas = FigureCanvasTkAgg(self.fig, master = self.window)  
            self.canvas.draw()

            # placing the canvas on the Tkinter window
            self.canvas.get_tk_widget().grid(row=0, column=2, columnspan=2, padx=10, pady=2)

        # self.window.after(1000, self.create_plot)
    
    def update_plot(self):

        if self.run_preds:
            # plotting the graph
            preds, times = self.format_preds_for_plot(self.pred_probs)
            self.make_plot(preds, times)
                
            self.canvas.draw_idle()

        self.window.after(1000, self.update_plot)
    
    def plot_final(self):
        pass        

    # BUTTONS
    def create_show_vid_button(self):
        self.show_vid_button = ctk.CTkButton(self.window, text="Show/Hide Video", command=self.switch_show_webcam)
        self.show_vid_button.grid(row=1, column=0, padx=10, pady=2, sticky='w')
    
    def create_run_preds_button(self):
        self.preds_button = ctk.CTkButton(self.window, text="Play/Pause Predictions", command=self.switch_run_preds)
        self.preds_button.grid(row=1, column=2, padx=10, pady=2, sticky='w')

    def create_end_session_button(self):
        self.end_session = ctk.CTkButton(self.window, text = "End Session", command=self.end_session)
        self.end_session.grid(row=1, column=3, padx=10, pady=2, sticky='e')

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

    def end_session(self):
        
        self.show_webcam = False
        self.video_frame.configure(image='')
        self.video_frame._image_cache = None
        self.run_preds = False
        self.stop_webcam()
        

        self.plot_final()

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