# ITMO University
# Mobile Computer Vision course
# 2020
# by Aleksei Denisov
# denisov@itmo.ru

import cv2
import numpy as np

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen




def write_gpio(f, v):
    if v:
        f.write('1')
        f.flush()
    else:
        f.write('0')
        f.flush()

def init_gpio():
    with open("/sys/class/gpio/export", 'w') as f:
        f.write("13")
        f.flush()
    with open("/sys/class/gpio/export", 'w') as f:
        f.write("19")
        f.flush()
    with open("/sys/class/gpio/export", 'w') as f:
        f.write("20")
        f.flush()
    with open("/sys/class/gpio/gpio13/direction", 'w') as f:
        f.write("out")
        f.flush()
    with open("/sys/class/gpio/gpio19/direction", 'w') as f:
        f.write("out")
        f.flush()
    with open("/sys/class/gpio/gpio20/direction", 'w') as f:
        f.write("out")
        f.flush()

def set_blue(value):
    with open("/sys/class/gpio/gpio13/value", 'w') as f:
        write_gpio(f, value)


def set_red(value):
    with open("/sys/class/gpio/gpio19/value", 'w') as f:
        write_gpio(f, value)


def set_green(value):
    with open("/sys/class/gpio/gpio20/value", 'w') as f:
        write_gpio(f, value)


def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


def check_color(img, lower, upper):
    return np.count_nonzero(cv2.inRange(img, lower, upper)) / (img.shape[0] * img.shape[1])


def show_camera():
    try:
        init_gpio()
    except:
        pass
    current_state = -1
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([25, 255, 255])

    lower_red2 = np.array([155, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=4))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        # window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while True:
            ret_val, frame = cap.read()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Show video
            recsize = 128
            cv2.imshow('Inverted', cv2.rectangle(frame, (1280 // 2 - recsize // 2, 720 // 2 - recsize // 2),
                                                 (1280 // 2 + recsize // 2, 720 // 2 + recsize // 2), (255, 0, 0), 5))

            cropped = hsv[720 // 2 - recsize // 2: 720 // 2 + recsize // 2,
                      1280 // 2 - recsize // 2: 1280 // 2 + recsize // 2]
            # cropped = cv2.convertScaleAbs(cropped, alpha=1.5, beta=0)
            # if check_color(cropped, lower_blue, upper_blue) >= 0.9:
            #     print("blue")
            #     print(check_color(cropped, lower_blue, upper_blue))
            #
            #     if current_state == 1:
            #         pass
            #     else:
            #         set_red(False)
            #         set_green(False)
            #         set_blue(True)
            #         current_state = 1
            # elif check_color(cropped, lower_green, upper_green) >= 0.9:
            #     print("green")
            #     if current_state == 2:
            #         pass
            #     else:
            #         set_red(False)
            #         set_green(True)
            #         set_blue(False)
            #         current_state = 2
            # elif check_color(cropped, lower_red1, upper_red1) + check_color(cropped, lower_red2, upper_red2) >= 0.9:
            #     print("red")
            #     if current_state == 3:
            #         pass
            #     else:
            #         set_red(True)
            #         set_green(False)
            #         set_blue(False)
            #         current_state = 3
            # else:
            #     if current_state == -1:
            #         pass
            #     else:
            #         set_red(False)
            #         set_green(False)
            #         set_blue(False)
            #         current_state = -1

            # This also acts as
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
