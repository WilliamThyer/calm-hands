import cv2
import pathlib
import os
import time
import uuid

class Cam():

    def __init__(self, base_folder = 'frames'):

        self.capture = cv2.VideoCapture(0)
        self.base_path = pathlib.Path(base_folder)

        if not os.path.isdir(self.base_path):
            os.mkdir(self.base_path)

    def get_image(self):
        
        _, frame = self.capture.read()
        
        return frame

    def make_labelled_folder(self, label='misc'):

        labelled_folder_path = self.base_path / label

        if not os.path.isdir(labelled_folder_path):
            os.mkdir(labelled_folder_path)

        return labelled_folder_path
    
    def write_frame_stream(self, label = 'misc', subtitle = '', length = 5, wait = 1):

        labelled_folder_path = self.make_labelled_folder(label)

        start_time = time.time()

        while time.time() < start_time + length:
            frame = self.get_image()
            cv2.imwrite(str(labelled_folder_path/f'{subtitle}{str(uuid.uuid4())}.png'), frame)
            time.sleep(wait)