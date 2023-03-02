# Calm Hands 🙌

By William Thyer, 2023

<img src="gui_screenshots/app_demo.gif" alt="drawing" width="800"/>

Calm Hands helps the user reduce nail-biting during computer use. It provides realtime feedback about nail-biting habits using a deep neural net that monitors images from your webcam stream. This process is entirely local and images are never saved. Feedback is provided through audio and visual cues to alert you of when you are biting your nails. Realtime data visualization is provided as well.

Skills: Deep learning, computer vision, data visualization, software development.

Built with: Fastai, OpenCV, Tkinter, CustomTkinter, Matplotlib.

## How I Made This

### Step 1. Collect training and heldout test images

First I had to collect several hundred images of my biting my nails and not biting my nails (but doing other things). So I created `camera.py` and call it in `collect_training_data.ipynb`. This allowed me to collect hundreds of photos in a variety of locations, lighting setups, and angles very easily.

### Step 2. Train the image classifier in Google Colab with fastai

I trained an `edgenext_small` model (imported from the `timm` library) using fastai. I used the proven method of finetuning a pretrained image classifier on this specific task. With ~1000 images and 3 cycles of training, I was at >90% accuracy. But I found there were specific positions and angles that the model was getting wrong.

### Step 3. Collecting more data based on model mistakes

I created the `realtime_model_preds.ipynb` to collect more data quickly, based on the predictions of the existing model. I moved around until I found positions that the model was incorrectly predicting. Then, I pressed either '1' or '2' on the keyboard to save that frame to the correct folder and add it to the training set.

After collecting several hundred more photos, I retrained the model. Now, accuracy was exceeding 98%.

### Step 4. Creating the app

I used `tkinter` and `customtkinter` to create the GUI for the app. It displays the webcam feed on one side, and displays an `matplotlib` plot of the predictions of the model. It also provides instant auditory feedback if I'm biting my nails.

## Files

### `app.py`

The primary Calm Hands app. Running this opens the app that monitors your nail biting habits in real time. Provides a webcam feed and data visualization of your behavior/model predictions.

### `camera.py`

I used this primarily during data collection. Contains a useful `Cam` class for collecting frames from the webcam to train the neural net.

### `collect_training_data.ipynb`

Main notebook for collecting training data for the deep neural net.

### `realtime_model_preds.ipynb`

Secondary notebook that contains an extremely useful interface for collecting training data for edge cases. You can get realtime model predictions, and then collect data for either class by pressing certain buttons. I used this to collect examples that were being mispredicted.

### `training_calm_hands_colab.ipynb`

Notebook that I ran in Google Colab to train the `edgenext_small` model. I chose `edgenext` because it is a smaller model that still has high performance.

### `dummy_model.ipynb`

A model that replicates the functionality of a fastai model, with random predictions. Useful for testing the `App` without loading a real model or importing fastai.
