import cv2
import numpy as np


# trackbar callback fucntion to update HSV value
def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    # assign trackbar position value to H,S,V High and low variable
    H_low = cv2.getTrackbarPos('low H', windowName)
    H_high = cv2.getTrackbarPos('high H',windowName)
    S_low = cv2.getTrackbarPos('low S',windowName)
    S_high = cv2.getTrackbarPos('high S',windowName)
    V_low = cv2.getTrackbarPos('low V',windowName)
    V_high = cv2.getTrackbarPos('high V',windowName)


# create a seperate winow namedwindowName for trackbar
windowName ='controls'
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)




# global variable
H_low = 0
H_high = 179
S_low = 0
S_high = 255
V_low = 0
V_high = 255

# create trackbars for high,low H,S,V
cv2.createTrackbar('low H',windowName, 0, 179, callback)
cv2.createTrackbar('high H',windowName, 179, 179, callback)

cv2.createTrackbar('low S',windowName, 0, 255, callback)
cv2.createTrackbar('high S',windowName, 255, 255, callback)

cv2.createTrackbar('low V',windowName, 0, 255, callback)
cv2.createTrackbar('high V',windowName, 255, 255, callback)

while (1):
    # read source image
    img = cv2.imread("blobs.jpg")
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
       detector = cv2.SimpleBlobDetector(params)
    else:
       detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # convert sourece image to HSC color mode
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    hsv_low = np.array([H_low, S_low, V_low])
    hsv_high = np.array([H_high, S_high, V_high])

    # making mask for hsv range
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    print(mask)
    # masking HSV value selected color becomes black
    res = cv2.bitwise_and(img, img, mask=mask)

    # show image
    cv2.imshow(windowName, mask)
    cv2.imshow(windowName, res)

    # waitfor the user to press escape and break the while loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# destroys all window

cv2.destroyAllWindows()