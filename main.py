import cv2

# define a video capture object
video = cv2.VideoCapture(0)

while(True):

    _, frame = video.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontal_face = cv2.CascadeClassifier(
        'Neural_Network/haarcascade_frontalface_default.xml')
    results = frontal_face.detectMultiScale(
        frame_gray, scaleFactor=1.2, minNeighbors=3)

    for(parameter_1, parameter_2, weight, height) in results:
        cv2.rectangle(frame, (parameter_1, parameter_2), (parameter_1 +
                      weight, parameter_2 + height), (140, 50, 20), thickness=7)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break

# Destroy all the windows
cv2.destroyAllWindows()
