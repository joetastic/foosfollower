import cv2, numpy as np, sys

cap = cv2.VideoCapture('Horton_Beine vs. Atha_Loffredo at the 2014 Tornado Worlds.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 1830)

MAX_BALL_AREA = 600

# Positioning
FIELD_CORNERS = (
    (153, 36), (456, 37),
    (74, 461), (518, 460)
)

ROD_YVALS = (122, 162, 203, 258, 320, 395, 487, 603)

# Thresholds
BALL_THRESHOLDS = ((0, 116, 25), (15, 255, 255))

GLARE_THRESHOLDS = ((0, 116, 0), (255, 255, 255))

FIELD_THRESHOLDS = ((0, 0, 0), (31, 255, 255))


def draw_rods(frame):
    for yval in ROD_YVALS:
        cv2.line(frame, (0, yval), (frame.shape[1], yval), (255, 0, 0), 2)
    return frame

def equalize(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    #v = cv2.equalizeHist(v)

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def find_ball_hough(frame, lower, upper, x):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #mask = cv2.inRange(hsv, BALL_THRESHOLDS[0], BALL_THRESHOLDS[1])

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0]/20, 30, 10, 5, 10, 20)
    for circle in circles[0]:
        print circle
        frame = cv2.circle(frame, tuple(circle[:2]), circle[2], (0, 0, 255))
    return frame

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > MAX_BALL_AREA:
            print 'perimeter/area:', cv2.arcLength(contour, True)/cv2.contourArea(contour)
    cv2.drawContours(frame, [c for c in contours
                             if cv2.contourArea(c) > MAX_BALL_AREA
                             and cv2.arcLength(c, True)/cv2.contourArea(c) < 0.14],
                     -1, (255, 0, 0), 4)
    return frame

def find_ball_threshold(frame, thresh):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    #mask = cv2.inRange(hsv, (0, 0, 13), (15, 255, 255))
    mask = cv2.inRange(hsv, (0, 140, 110), (187, 197, 124))
    cv2.rectangle(mask, (250, 257), (372, 324), 0, -1)
    return mask

def get_ball_template():
    template = np.zeros((22, 22), dtype=np.uint8)
    cv2.circle(template, (11, 11), 11, 255, -1)
    return template

def match_template(mask):
    method = cv2.TM_CCOEFF
    method = cv2.TM_SQDIFF
    method = cv2.TM_CCORR
    res = cv2.matchTemplate(mask, get_ball_template(), method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + 22, top_left[1] + 22)
    return top_left, bottom_right

def remove_glare(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GLARE_THRESHOLDS[0], GLARE_THRESHOLDS[1])
    return cv2.bitwise_and(frame, frame, mask=mask)

def remove_field(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIELD_THRESHOLDS[0], FIELD_THRESHOLDS[1])
    return cv2.bitwise_and(frame, frame, mask=mask)

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

def draw_board(frame):
    frame = cv2.warpPerspective(frame, get_board_mtx(frame),
                                tuple(reversed(frame.shape[:2])))
    return frame

# NOTE - to write out, use 'jpeg'

def main():
    counter = 1830
    cv2.namedWindow('image')
    cv2.moveWindow('image', 0, 0)
    def mouse(event, x, y, *args):
        if event == 1:
            print 'mouse: ', x, y
    while True:
        ret, orig = cap.read()
        counter += 1
        print counter
        def render(vals):
            H = cv2.getTrackbarPos('H', 'image')
            S = cv2.getTrackbarPos('S', 'image')
            V = cv2.getTrackbarPos('V', 'image')

            frame = orig.copy()
            #frame = draw_rods(frame)
            #frame = draw_board(frame)
            #frame = equalize(frame)
            #frame = remove_glare(frame)
            #frame = remove_field(frame)
            mask = find_ball_threshold(draw_board(frame), (H, S, V))
            top_left, bottom_right = match_template(mask)
            mtx = get_board_mtx(frame)
            t = cv2.perspectiveTransform(
                np.array([[top_left, bottom_right]], dtype='float32'),
                cv2.invert(mtx)[1])
            top_left, bottom_right = t[0]
            cv2.rectangle(frame, tuple(top_left), tuple(bottom_right),
                          255, 1)

            #frame = cv2.bitwise_and(frame, frame, mask=mask)
            print H, S, V
            #frame = find_ball_threshold(frame, (H, S, V))
            #frame = np.concatenate((frame, orig), axis=0)
            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            cv2.imshow('image', frame)
        cv2.setMouseCallback('image', mouse)
        hsvinit = (0, 140, 110)
        cv2.createTrackbar('H','image',hsvinit[0],255,render)
        cv2.createTrackbar('S','image',hsvinit[1],255,render)
        cv2.createTrackbar('V','image',hsvinit[2],255,render)
        render(None)

        cv2.waitKey(1)

if __name__ == '__main__':
    main()
