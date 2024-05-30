import cv2

# Context manager version of OpenCV's video capture
class ConVideoCapture(cv2.VideoCapture):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
