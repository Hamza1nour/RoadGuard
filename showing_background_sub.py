import cv2


cap = cv2.VideoCapture('video_vehicles.mp4')


algo = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (1020, 500))

    
    mask = algo.apply(frame)

    
    cv2.imshow('Black-and-White Frame', mask)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
