import cv2
import numpy as np

# Capturing the video
vid = cv2.VideoCapture("C:\\Users\\Argha Kamal Samanta\\OneDrive\\Desktop\\default.mp4")

while True:
    ret, frame = vid.read()

    # Defining a temporary variable for further reference
    temp = np.copy(frame)

    # Using ret = FALSE condition to close the video window after completion of the video
    if not ret:
        break

    # Making the " frame " a grayscale image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Using Gaussian blur for noise cancellation
    cv2.GaussianBlur(frame, (5, 5), 0, frame, 0, 0)

    # Applying canny edge detection
    frame = cv2.Canny(frame, 20, 150)

    # Defining ROI by creating a triangle in which the correct lane is present with most probability
    h, w = frame.shape
    triangle = np.array([
            [((50 * w) // 293, h), ((130 * w) // 293, (6.5 * h) // 16), ((250 * w) // 293, h)]
    ])
    # Making an image of size same with " frame " with pixel values = 0
    mask = np.zeros_like(frame)
    # Making a white triangle of size assigned before with a black background
    mask = cv2.fillPoly(mask, np.int32(triangle), 255)
    # Making the region outside the triangle black in " frame "
    frame = cv2.bitwise_and(frame, mask)

    # Finding straight lines in the updated " frame "
    lines = cv2.HoughLinesP(frame, 1, np.pi / 180, 50, None, 15, 10)

    # Making " frame " 3 channeled so that it can be added to the 3 channeled " temp "
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Drawing the lines in red colour in the " frame " itself
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3)

    # Adding the lines detected to the real image
    frame = cv2.addWeighted(temp, 0.8, frame, 1.0, 1.0)

    # Displaying the final " frame "
    cv2.imshow("Detecting the correct lane....", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
