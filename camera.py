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
    
    def write_image(self,labelled_folder_path,subtitle):
        
        frame = self.get_image()
        file_name = str(labelled_folder_path/f'{subtitle}{str(time.time()).replace(".",",")}.png')
        cv2.imwrite(file_name, frame)

    def make_labelled_folder(self, path='frames/misc'):

        labelled_folder_path = pathlib.Path(path)

        if not os.path.isdir(labelled_folder_path):
            os.makedirs(labelled_folder_path)

        return labelled_folder_path
    
    def write_frame_stream(self, path = 'frames/misc', subtitle = '', length = 5, wait = 1, show_cam=True):

        labelled_folder_path = self.make_labelled_folder(path)

        start_time = time.time()
        while time.time() < start_time + length:
            
            self.write_image(labelled_folder_path, subtitle)
            time.sleep(wait)

            if show_cam:
                self.show_webcam()
                if self.check_close_cam():
                    cv2.destroyAllWindows()
                    break
        
        if show_cam:
            cv2.destroyAllWindows()

    def show_webcam(self):

        _, img = self.capture.read()
        img = cv2.flip(img, 1)
        cv2.imshow('Cam', img)

    def check_close_cam(self):
        
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return True
        if cv2.getWindowProperty('Cam',cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return True