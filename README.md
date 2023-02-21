# Calm Hands 🙌

By William Thyer, 2023

<img src="gui_screenshots/demo.gif" alt="drawing" height="400"/>

Calm Hands helps the user reduce nail-biting during computer use. It provides realtime feedback about nail-biting habits using a deep neural net that monitors images from your webcam stream. This process is entirely local and images are never saved. Feedback is provided through audio and visual cues to alert you of when you are biting your nails. Realtime data visualization is provided as well.

Skills: Deep learning, computer vision, data visualization, software development.

Built with: Fastai, OpenCV, tkinter, Matplotlib.

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

Notebook that I ran in Google Colab to train the `Edgenext` model.
