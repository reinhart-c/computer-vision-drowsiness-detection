import cv2

class FrameExtractor:
    def extract_frames(self):
        feed = cv2.VideoCapture(0)

        i = 0
        while True:
            ret, frame = feed.read()
            if not ret or frame is None:
                continue
            if i%3 == 0:
                yield frame
                i = 0
            i += 1