import cv2 as cv
import mediapipe as mp


def nothing_for_trackbar(x):
    pass


cv.namedWindow('Controls')
cv.createTrackbar('Hand', 'Controls', 0, 1, nothing_for_trackbar)
cv.createTrackbar('Laplacian', 'Controls', 0, 2, nothing_for_trackbar)
cv.createTrackbar('Blur', 'Controls', 0, 100, nothing_for_trackbar)

vid = cv.VideoCapture(0)
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
writer = cv.VideoWriter('basic-video.avi', cv.VideoWriter_fourcc(*'DIVX'),
                        20, (width, height))

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    ret, frame = vid.read()

    """ Motion detection """
    hand_on_off = int(cv.getTrackbarPos('Hand', 'Controls'))
    if hand_on_off == 0:
        pass
    elif hand_on_off == 1:
        results = hands.process(frame)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # if id ==0:
                    cv.circle(frame, (cx, cy), 3, (255, 0, 255), cv.FILLED)
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    """ Laplacian gradient """
    laplace_on_off = int(cv.getTrackbarPos('Laplacian', 'Controls'))
    if laplace_on_off == 0:
        pass
    elif laplace_on_off == 1:
        frame = cv.Laplacian(frame, cv.CV_8U)
    elif laplace_on_off == 2:
        frame = cv.Laplacian(frame, cv.CV_64F)

    """ Blur """
    blur_intensity = int(cv.getTrackbarPos('Blur', 'Controls'))
    frame = cv.blur(frame, (blur_intensity + 1, blur_intensity + 1))

    writer.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
writer.release()
cv.destroyAllWindows()
