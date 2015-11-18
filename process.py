import cv2, numpy as np

FIELD_CORNERS = (
    (153, 36), (456, 37),
    (74, 461), (518, 460)
)

def get_ball_template():
    template = np.zeros((22, 22), dtype=np.uint8)
    cv2.circle(template, (11, 11), 11, 255, -1)
    return template

def match_template(mask):
    res = cv2.matchTemplate(mask, get_ball_template(), cv2.TM_CCORR)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + 22, top_left[1] + 22)
    return top_left, bottom_right

def process(frame):
    frame = cv2.warpPerspective(frame, get_board_mtx(frame),
                                tuple(reversed(frame.shape[:2])))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(hsv, (0, 140, 110), (187, 197, 124))
    top_left, bottom_right = match_template(mask)
    cv2.rectangle(frame, tuple(top_left), tuple(bottom_right),
                  255, 1)
    print top_left, bottom_right
    return frame

def get_board_mtx(frame):
    dest = np.zeros(frame.shape)
    corners = [
        (0, 0),
        (dest.shape[1], 0),
        (0, dest.shape[0]),
        (dest.shape[1], dest.shape[0]),
    ]
    mtx = cv2.getPerspectiveTransform(np.array(FIELD_CORNERS).astype(np.float32),
                                      np.array(corners).astype(np.float32))
    return mtx


def main():
    cap = cv2.VideoCapture('Horton_Beine vs. Atha_Loffredo at the 2014 Tornado Worlds.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1830)

    cv2.namedWindow('image')
    cv2.moveWindow('image', -200, -1000)

    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('image', frame)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
