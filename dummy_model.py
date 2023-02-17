# create class DummyModel
import numpy as np

class DummyModel():

    def __init__(self, classes=['Biting','Not biting']):

        self.classes = classes
        self.indices = np.arange(len(self.classes))

    def predict(self, image, no_bar=True, no_logging=True):

        pred = np.random.choice(self.indices)
        pred_class = self.classes[pred]
        prob = np.random.rand()
        probs = [prob, 1-prob]

        return pred_class, pred, probs 