import ctypes
import cv2
import numpy as np
import time


def draw_result(frame, n, boxes):
    for i in range(n):
        x = boxes[6*i + 0]
        y = boxes[6*i + 1]
        w = boxes[6*i + 2]
        h = boxes[6*i + 3]
        n = boxes[6*i + 4]
        p = boxes[6*i + 5]

        cv2.putText(frame, str(int(n)), (int(x - w/2), int(y - h/2-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, str(p), (int(x + w/2 - 25), int(y - h/2-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(frame, (int(x - w/2), int(y-h/2)),  (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)


def main():
    libc = ctypes.cdll.LoadLibrary("build/libyolov7.so")
    libc.loadEngine(ctypes.create_string_buffer("yolov7-tiny.engine".encode("utf8")))

    image = cv2.imread("image/test.jpeg")
    h, w = image.shape[:2]
    result = np.zeros((1000*6, ), dtype=np.float32)
    start = time.time()
    # bgr
    num = libc.inferImage(image.ctypes.data_as(ctypes.c_char_p), w, h, result.ctypes.data_as(ctypes.c_char_p))
    print("infer time: ", time.time() - start)
    draw_result(image, num, result)

    cv2.imwrite("result.jpg", image)
    libc.release()


def test_fps():
    capture = cv2.VideoCapture(0)
    libc = ctypes.cdll.LoadLibrary("build/libyolov7.so")
    libc.loadEngine(ctypes.create_string_buffer("yolov7-tiny.engine".encode("utf8")))
    while True:
        ref, frame = capture.read()
        t1 = time.time()
        h, w = frame.shape[:2]
        result = np.zeros((1000*6, ), dtype=np.float32)
        num = libc.inferImage(frame.ctypes.data_as(ctypes.c_char_p), w, h, result.ctypes.data_as(ctypes.c_char_p))
        draw_result(frame, num, result)
        t2 = time.time()
        frame = cv2.putText(frame, "time = %.4f" % (t2-t1), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", frame)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # main()
    test_fps()
