import glob
import os
import sys
import time
import cv2
import csv
import numpy as np


# initialization
floor_position = "right"
ONLINE = True
CALIBRATE = False
# path for the result saving
file_name = "Scene1.mp4"
RELATIVE_DESTINATION_PATH = file_name + 'result/'
RELATIVE_TRUTH_PATH = 'truth_rat/'
FPS = 60
THRESHOLD_WALL_VS_FLOOR = 80
THRESHOLD_ANIMAL_VS_FLOOR = 55
HD = 1280, 640
BGR_COLOR = {'red': (0, 0, 255),
             'green': (127, 255, 0),
             'blue': (255, 127, 0),
             'yellow': (0, 127, 255),
             'black': (0, 0, 0),
             'white': (255, 255, 255)}
WAIT_DELAY = 1

# for cropping
perspectiveMatrix = dict()
croppingPolygon = np.array([[0, 0]])
croppingPolygons = dict()
rectangle = []
name = ""

RENEW_TETRAGON = True


def counterclockwiseSort(rectangle):
    rectangle = sorted(rectangle, key=lambda e: e[0])
    rectangle[0:2] = sorted(rectangle[0:2], key=lambda e: e[1])
    rectangle[2:4] = sorted(rectangle[2:4], key=lambda e: e[1], reverse=True)
    return rectangle

# mouse callback function for drawing a cropping polygon
def drawFloorCrop(event, x, y, flags, params):
    global perspectiveMatrix, name, RENEW_TETRAGON
    imgCroppingPolygon = np.zeros_like(params['imgFloorCorners'])
    if event == cv2.EVENT_RBUTTONUP:
        cv2.destroyWindow(f'Floor Corners for {name}')
    if len(params['croppingPolygons'][name]) > 4 and event == cv2.EVENT_LBUTTONUP:
        RENEW_TETRAGON = True
        h = params['imgFloorCorners'].shape[0]
        # delete 5th extra vertex of the floor cropping rectangle
        params['croppingPolygons'][name] = np.delete(params['croppingPolygons'][name], -1, 0)
        params['croppingPolygons'][name] = params['croppingPolygons'][name] - [h, 0]

        # Sort cropping rectangle vertices counter-clockwise starting with top left
        params['croppingPolygons'][name] = counterclockwiseSort(params['croppingPolygons'][name])
        # Get the matrix of perspective transformation
        params['croppingPolygons'][name] = np.reshape(params['croppingPolygons'][name], (4, 2))
        rectVertices = np.float32(params['croppingPolygons'][name])
        tetragonVerticesUpd = np.float32([[0, 0], [0, h], [h, h], [h, 0]])
        perspectiveMatrix[name] = cv2.getPerspectiveTransform(rectVertices, tetragonVerticesUpd)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON:
            params['croppingPolygons'][name] = np.array([[0, 0]])
            RENEW_TETRAGON = False
        if len(params['croppingPolygons'][name]) == 1:
            params['croppingPolygons'][name][0] = [x, y]
        params['croppingPolygons'][name] = np.append(params['croppingPolygons'][name], [[x, y]], axis=0)
    if event == cv2.EVENT_MOUSEMOVE and not (len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON):
        params['croppingPolygons'][name][-1] = [x, y]
        if len(params['croppingPolygons'][name]) > 1:
            cv2.fillPoly(
                imgCroppingPolygon,
                [np.reshape(
                    params['croppingPolygons'][name],
                    (len(params['croppingPolygons'][name]), 2)
                )],
                BGR_COLOR['green'], cv2.LINE_AA)
            imgCroppingPolygon = cv2.addWeighted(params['imgFloorCorners'], 1.0, imgCroppingPolygon, 0.5, 0.)
            cv2.imshow(f'Floor Corners for {name}', imgCroppingPolygon)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return np.abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def floorCrop(file_name):
    global perspectiveMatrix, rectangle, name, croppingPolygons, floor_position
    name = os.path.splitext(file_name)[0]
    cap = cv2.VideoCapture(file_name)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    if floor_position == "right":
        frame = frame[:, w - h: w]
    elif floor_position == "left":
        frame = frame[:, 0: h]
    elif floor_position == "mid":
        frame = frame[:, (w - h) / 2: (w + h) / 2]
    else:
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
            max_cos = np.max([angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
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

    rectVertices = counterclockwiseSort(rectVertices)
    croppingPolygons[name] = rectVertices
    rectVertices = np.float32(rectVertices)
    tetragonVerticesUpd = np.float32([[0, 0], [0, h], [h, h], [h, 0]])
    perspectiveMatrix[name] = cv2.getPerspectiveTransform(np.float32(croppingPolygons[name]), tetragonVerticesUpd)
    frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (h, h))

    # show both floor frame and gray new frame[h*h] with contour
    imgFloorCorners = np.hstack([frame, frameGray])
    cv2.imshow(f'Floor Corners for {name}', imgFloorCorners)
    k = cv2.waitKey(0)
    if k == 27:
        sys.exit()
    cv2.destroyWindow(f'Floor Corners for {name}')
    return rectVertices, perspectiveMatrix[name]


def trace(file_name):
    global perspectiveMatrix, croppingPolygons, rectangle, name, WAIT_DELAY
    # initialize
    run_time = 120

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

    video = cv2.VideoWriter(f'{RELATIVE_DESTINATION_PATH}timing/{name}_trace.avi',
                            cv2.VideoWriter_fourcc(*'X264'),
                            FPS, HD, cv2.INTER_LINEAR)
    imgTrack = np.zeros_like(frame)

    start = time.time()
    distance = old_x = old_y = 0
    total_error = 0
    max_dis_per_frame = 0

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

        # pre-process the frame to find the contour of animal
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("pre-processing grayscale", frameGray)
        kernelSize = (25, 25)
        frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
        # cv2.imshow("pre-processing blur", frameBlur)
        _, thresh = cv2.threshold(frameBlur, THRESHOLD_ANIMAL_VS_FLOOR, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) < 1:
            continue

        # Find a contour with the biggest area (animal most likely)
        contour = contours[np.argmax(list(map(cv2.contourArea, contours)))]
        # find the center point of the animal
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        if old_x == 0 and old_y == 0:
            old_x = x
            old_y = y
        dis_per_frame = np.sqrt(((x - old_x) / float(h)) ** 2 + ((y - old_y) / float(h)) ** 2)
        distance += dis_per_frame
        if dis_per_frame > max_dis_per_frame:
            max_dis_per_frame = dis_per_frame
            # print('max speed is:', max_dis_per_frame)

        if ONLINE:
            # Draw the most acute angles of the contour (tail/muzzle/paws of the animal)
            #cv2.imshow("Otsu result", thresh)
            #cv2.waitKey(0)
            hull = cv2.convexHull(contour)
            imgPoints = np.zeros(frame.shape, np.uint8)
            for i in range(2, len(hull) - 2):
                if np.dot(hull[i][0] - hull[i - 2][0], hull[i][0] - hull[i + 2][0]) > 0:
                    imgPoints = cv2.circle(imgPoints, (hull[i][0][0], hull[i][0][1]), 5, BGR_COLOR['yellow'], -1,
                                           cv2.LINE_AA)

            # Draw a contour and a centroid of the animal
            cv2.drawContours(imgPoints, [contour], 0, BGR_COLOR['green'], 2, cv2.LINE_AA)
            imgPoints = cv2.circle(imgPoints, (x, y), 5, BGR_COLOR['black'], -1)

            # Draw a track line of the animal movement
            imgTrack = cv2.addWeighted(np.zeros_like(imgTrack), 0.85, cv2.line(imgTrack, (x, y), (old_x, old_y),
                                                                               (255, 127, int(cap.get(
                                                                                   cv2.CAP_PROP_POS_AVI_RATIO) * 255)),
                                                                               1, cv2.LINE_AA), 0.98, 0.)
            imgContour = cv2.add(imgPoints, imgTrack)

            # left image
            frame = cv2.bitwise_and(frame, frame, mask=thresh)
            frame = cv2.addWeighted(frame, 0.5, imgContour, 1.0, 0.)

            speed = dis_per_frame * 10

            # text of the information
            cv2.putText(frame, 'Distance ' + str('%.2f' % distance),
                        (150, 260), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['white'])
            cv2.putText(frame, 'Speed ' + str('%.5f /sec' % speed),
                        (150, 290), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['white'])
            cv2.putText(frame, 'Time ' + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)),
                        (150, 320), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['white'])
            cv2.circle(frame, (x, y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)

            # show both of contour frame and color frame
            layout = np.hstack((frame, frameColor))
            cv2.imshow(f'Open Field Trace of {name}', layout)

            # write the video to the certain fold
            video.write(cv2.resize(layout, HD))

            k = cv2.waitKey(WAIT_DELAY) & 0xff
            if r_time >= run_time+0.1:
                break
            if k == 27:
                break
            if k == 32:
                if WAIT_DELAY == 1:
                    WAIT_DELAY = 0  # pause
                else:
                    WAIT_DELAY = 1  # play as fast as possible
        # record information every 2 seconds

        if r_time % 2 == 0:
            error, total_error = computeError(h, x, y, r_time, total_error)

            file = open(RELATIVE_DESTINATION_PATH + 'results.csv', 'a')
            file.write(name + ',%.1f' % r_time + ',%.2f' % distance + ',%.5f' % speed +
                       ',%.1f' % x + ',%.1f' % y + ',%.4f' % error + ',%.4f\n' % total_error)
            file.close()

        old_x = x
        old_y = y
    cv2.destroyAllWindows()
    cap.release()

    if ONLINE:
        video.release()
        cv2.imwrite(RELATIVE_DESTINATION_PATH + 'traces/' + name + '_[distance]=%.2f' % distance +
                    '_[time]=%.1fs' % r_time + '.png', cv2.resize(imgTrack, (max(HD), max(HD))))

    print(file_name + '\tdistance %.2f\t' % distance + 'processing/real time %.1f' % float(
       time.time() - start) + '/%.1f s' % r_time)


def computeError(h, x, y, r_time, total_error):
    error = 0
    # print("At time:", r_time, "the position is", [x, y])

    with open(RELATIVE_TRUTH_PATH + "truth.csv", "r") as csvFile:
        reader = csv.reader(csvFile)
        rows = [row for row in reader]

    for time_n in range(len(rows)):
        if rows[time_n][0] == str(r_time):
            # print('The true position value of the animal is:', rows[time_n])
            # print('true x position:', rows[time_n][1], 'true y position:', rows[time_n][2])
            diff_x = (x - float(rows[time_n][1])) / float(h)
            diff_y = (y - float(rows[time_n][2])) / float(h)
            error = np.sqrt(diff_x ** 2 + diff_y ** 2)
            total_error += error
            print('Sqaure error in time', r_time, 'is ', error, 'total error is:', total_error)

    return error, total_error


def creatCSV():
    # create "timing" and "traces" folds
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'traces'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'traces')
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'timing'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'timing')
    # create "results.csv" and initial with some attributes
    file = open(RELATIVE_DESTINATION_PATH + 'results.csv', 'w')
    file.write('Animal,Time(s),Distance(unit of box side),Speed(unit/s),x,y,Error,Total Error\n')
    file.close()


if len(sys.argv) > 1 and '--online' in sys.argv:
    ONLINE = True

creatCSV()

floorCrop(file_name)
trace(file_name)
