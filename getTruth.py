import os
import sys
import cv2
import numpy as np

x_ps = []
y_ps = []
run_time = 120

# initialization
ONLINE = True
CALIBRATE = False
HD = 1280, 640
BGR_COLOR = {'red': (0, 0, 255),
             'green': (127, 255, 0),
             'blue': (255, 127, 0),
             'yellow': (0, 127, 255),
             'black': (0, 0, 0),
             'white': (255, 255, 255)}
WAIT_DELAY = 1
THRESHOLD_WALL_VS_FLOOR = 80
layout = np.zeros(0)
RELATIVE_TRUTH_PATH = 'truth_rat/'


def counterclockwiseSort(rectangle):
    rectangle = sorted(rectangle, key=lambda e: e[0])
    rectangle[0:2] = sorted(rectangle[0:2], key=lambda e: e[1])
    rectangle[2:4] = sorted(rectangle[2:4], key=lambda e: e[1], reverse=True)
    return rectangle


def angleCos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return np.abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# initialize for cropping
perspectiveMatrix = dict()
rectangle = []
name = ""
croppingPolygon = np.array([[0, 0]])
croppingPolygons = dict()


def floorCrop(file_name):
    global perspectiveMatrix, rectangle, name, croppingPolygons
    name = os.path.splitext(file_name)[0]
    cap = cv2.VideoCapture(file_name)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    frame = frame[:, w - h: w]

    # Convert to the gray video
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blurred frame
    kernelSize = (5, 5)
    frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
    # threshold frame
    retval, mask = cv2.threshold(frameBlur, THRESHOLD_WALL_VS_FLOOR, 255, cv2.THRESH_BINARY_INV)
    # find contours in the threshold frame
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rectangle = []
    HALF_AREA = 0.5 * h * h
    for contour in contours:
        contourPerimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)

        # If the contour is convex rectangle and its area is above a half of total frame area,
        # then it's most likely the floor
        if len(contour) == 4 and cv2.contourArea(contour) > HALF_AREA:
            contour = contour.reshape(-1, 2)
            max_cos = np.max([angleCos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
            if max_cos < 0.3:
                rectangle.append(contour)

    # Draw the floor contour to new frame
    frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)
    imgSquare = np.zeros_like(frameGray)
    cv2.fillPoly(imgSquare, rectangle, BGR_COLOR['red'], cv2.LINE_AA)
    # cv2.add(frameGray, imgSquare / 2, frameGray)
    cv2.drawContours(frameGray, rectangle, -1, BGR_COLOR['red'], 2, cv2.LINE_AA)

    # if there is no suitable floor, then just use the new frame[h*h] as the floor
    if len(rectangle) > 0:
        rectVertices = rectangle[0]
    else:
        rectVertices = np.float32([[0, 0], [0, h], [h, h], [h, 0]])

    # Sort the cropping rectangle vertices according to the following order:
    # [left,top], [left,bottom], [right,bottom], [right,top]
    rectVertices = counterclockwiseSort(rectVertices)
    croppingPolygons[name] = rectVertices
    rectVertices = np.float32(rectVertices)
    tetragonVerticesUpd = np.float32([[0, 0], [0, h], [h, h], [h, 0]])
    perspectiveMatrix[name] = cv2.getPerspectiveTransform(np.float32(croppingPolygons[name]), tetragonVerticesUpd)
    frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (h, h))

    # show both floor frame and gray new frame[h*h] with contour
    imgFloorCorners = np.hstack([frame, frameGray])
    cv2.imshow(f'Floor Corners for {name}', imgFloorCorners)
    # cv2.setMouseCallback(
    #     f'Floor Corners for {name}',
    #     drawFloorCrop,
    #     {'imgFloorCorners': imgFloorCorners, 'croppingPolygons': croppingPolygons},
    #     )
    k = cv2.waitKey(0)
    if k == 27:
        sys.exit()
    cv2.destroyWindow(f'Floor Corners for {name}')
    return rectVertices, perspectiveMatrix[name]


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_ps.append(x)
        y_ps.append(y)
        print('choosed:', x, y)
    return x_ps, y_ps


def set_truth(file_name):
    global perspectiveMatrix, croppingPolygons, rectangle, name, WAIT_DELAY, layout, run_time
    # croppingPolygons[name] = np.array([[0,0]])
    name = os.path.splitext(file_name)[0]
    cap = cv2.VideoCapture(file_name)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    background = frame.copy()
    i_frame = 1
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame is not None:
        ret, frame = cap.read()
        if frame is None:
            break
        background = cv2.addWeighted(frame, 0.5 * (1 - i_frame / n_frames),
                                     background, 0.5 * (1 + i_frame / n_frames), 0)
        i_frame += 1
    cap = cv2.VideoCapture(file_name)
    ret, frame = cap.read()

    frame = frame[:, w - h: w]

    while frame is not None:
        ret, frame = cap.read()

        if frame is None:  # not logical
            break
        frameColor = frame[:, w - h: w].copy()
        frame = cv2.subtract(frame, background)

        r_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        frame = frame[:, w - h: w]
        if len(croppingPolygons[name]) == 4:
            cv2.drawContours(frameColor, [np.reshape(croppingPolygons[name], (4, 2))], -1, BGR_COLOR['red'], 2,
                             cv2.LINE_AA)
        else:
            cv2.drawContours(frameColor, rectangle, -1, BGR_COLOR['red'], 2, cv2.LINE_AA)

        frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (h, h))

        cv2.putText(frame, 'Time ' + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)),
                    (200, 450), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['white'])

        if ONLINE:
            layout = np.hstack((frame, frameColor))
            cv2.imshow(f'Open Field Trace of {name}', layout)

            if r_time % 2 == 0:
                print("At time:", r_time, "the position is")
                cv2.namedWindow("Key frame")
                cv2.setMouseCallback("Key frame", on_EVENT_LBUTTONDOWN)
                cv2.imshow("Key frame", layout)
                cv2.waitKey(0)
                if x_ps[-1] is not None:
                    file = open(RELATIVE_TRUTH_PATH + 'truth.csv', 'a')
                    file.write(str(r_time) + ',%.1f' % x_ps[-1] + ',%.1f\n' % y_ps[-1])
                    file.close()
                    print("x position:", x_ps[-1], "y position", y_ps[-1])
                cv2.destroyWindow("Key frame")

            k = cv2.waitKey(WAIT_DELAY) & 0xff
            if r_time >= run_time:
                break
            if k == 27:
                break
            if k == 32:
                if WAIT_DELAY == 1:
                    WAIT_DELAY = 0  # pause
                else:
                    WAIT_DELAY = 1  # play as fast as possible
    cv2.destroyAllWindows()
    cap.release()


if not os.path.exists(RELATIVE_TRUTH_PATH):
    os.makedirs(RELATIVE_TRUTH_PATH)
file = open(RELATIVE_TRUTH_PATH + 'truth.csv', 'w')
file.write('key frame(second), x position, y position\n')
file.close()
# crop the floor
# for file_name in glob.glob('*.mp4'):
#     floorCrop(file_name)
# for file_name in glob.glob('*.mp4'):
#     set_truth(file_name)
# print("collecting end.", x_ps, y_ps)

file_name = "rat.mp4"
floorCrop(file_name)
set_truth(file_name)