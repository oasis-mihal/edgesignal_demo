from typing import Tuple

import numpy as np
from norfair import Detection

# Encapsulates the yolo bounding box and gender prediction
class YoloDetection:

    def __init__(self, confidence: float, pred_positions: np.ndarray, resize: Tuple[float, float]):
        """

        Args:
            confidence: Predicted confidence of the box containing a person
            pred_positions: First four rows of the YoloV9 prediction array (hold the x1,y1,w,h)
            resize: How much the images are scaled from the original size
        """
        self._confidence = confidence
        self._gender = ""
        self._gender_confidence = -1.0

        self._w = pred_positions[2]
        self._h = pred_positions[3]
        # Account for bounding boxes being centered on the object
        self._tlx = pred_positions[0] - self._w * 0.5
        self._tly = pred_positions[1] - self._h * 0.5

        # Rescale to original image size
        self._w = round(self._w * resize[0])
        self._h = round(self._h * resize[1])
        self._tlx = round(self._tlx * resize[0])
        self._tly = round(self._tly * resize[1])

    @property
    def confidence(self):
        return self._confidence

    def set_confidence(self, confidence):
        self._confidence = confidence

    @property
    def gender(self) -> str:
        return self._gender

    @property
    def gender_confidence(self) -> float:
        return self._gender_confidence

    def set_gender(self, gender_confidences: np.ndarray):
        if gender_confidences[0] > gender_confidences[1]:
            self._gender = "man"
        else:
            self._gender = "woman"

        self._gender_confidence = max(gender_confidences)

    @property
    def center_x(self):
        return self._tlx + self._w / 2

    @property
    def tlx(self):
        return self._tlx

    @property
    def tly(self):
        return self._tly

    @property
    def brx(self):
        return self._tlx + self._w

    @property
    def bry(self):
        return self._tly + self._h

    def as_norfair(self) -> Detection:
        bbox = np.array(
            [
                [self.tlx, self.tly],
                [self.brx, self.bry]
            ]
        )
        scores = np.array(
            [
                self._confidence,
                self._confidence
            ]
        )
        # Zero is human
        detect = Detection(points=bbox, scores=scores, label=0, data=self.gender)
        detect.crossed_line = False
        return detect
