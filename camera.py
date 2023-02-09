import cv2
import pathlib
import os
import time

class Cam():

    def __init__(self):

        self.capture = cv2.VideoCapture(0)

    def get_image(self):
        
        _, frame = self.capture.read()
        
        return frame

    def make_labelled_folder(self, path='frames/misc'):

        labelled_folder_path = pathlib.Path(label)

        if not os.path.isdir(labelled_folder_path):
            os.mkdir(labelled_folder_path)

        return labelled_folder_path
    
    def write_frame_stream(self, path = 'frames/misc', subtitle = '', length = 5, wait = 1):

        labelled_folder_path = self.make_labelled_folder(path)

        start_time = time.time()

        while time.time() < start_time + length:
            frame = self.get_image()
            file_name = str(labelled_folder_path/f'{subtitle}{str(time.time()).replace(".",",")}.png')
            cv2.imwrite(file_name, frame)
            time.sleep(wait)
    
    def show_webcam(self, mirror=False):

        while True:
            _, img = self.capture.read()
            img = cv2.flip(img, 1)
            cv2.imshow('Cam', img)

            if cv2.waitKey(1) == 27: 
                break  # esc to quit
            if cv2.getWindowProperty('Cam',cv2.WND_PROP_VISIBLE) < 1:        
                break # click X to close   
        
        cv2.destroyAllWindows()