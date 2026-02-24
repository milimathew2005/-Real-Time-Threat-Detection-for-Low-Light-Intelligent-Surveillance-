import cv2

def get_video_stream(source=0):
    """
    source = 0 for webcam
    source = path/to/video.mp4 for CCTV footage
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Error opening video stream")

    return cap