## tkinter app that opens a window and displays your webcam feed

import tkinter as tk
import cv2
import PIL.Image as Image, PIL.ImageTk as ImageTk
import time
import numpy as np

class App:
    
    def __init__(self):
        # initialize variables
        self.cap = None
        self.window = None
        self.video_frame = None
        self.text = None
        self.button = None
        self.button2 = None
        self.button4 = None
        self.pred_probs = []
        
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
    
    def add_text(self):
        self.text = tk.Label(self.window, text="Hello World", fg="black", bg="white")
        self.text.pack()
        
    def update_text(self,text):
        self.text.configure(text=text)
    
    def update_prediction(self):
        return np.random.choice(["Biting Nails","Not Biting Nails"]), np.random.random(1)

    def show_frame(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        frame = Image.fromarray(frame) # convert to PIL image
        frame = ImageTk.PhotoImage(frame) # convert to PhotoImage
        self.video_frame.configure(image=frame) # display the image
        self.video_frame._image_cache = frame # avoid garbage collection
        self.window.after(15, self.show_frame) # 15 ms delay

        pred,pred_prob = self.update_prediction()
        self.pred_probs.append(pred_prob)

        self.update_text(pred)
        self.update_chart()
        
    def start_webcam(self):
        # start webcam feed
        print(self.cap) 
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
    app = App()
    app.run() 