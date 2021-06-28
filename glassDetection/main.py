# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:20:37 2018
based on the work was done by James Wu
"""

import dlib
import cv2
import numpy as np


# ==============================================================================
#   1. Landmarks format conversion function
# # Input: landmarks in dlib format
# # Output: landmarks in numpy format
# ==============================================================================
def landmarks_to_np(landmarks, dtype="int"):
    # Get the number of landmarks
    num = landmarks.num_parts

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def generate_np_landmarks(original_landmarks, data_type):
    num_of_landmarks = original_landmarks.num_parts
    coordinates = np.zeros((num_of_landmarks, 2), dtype=data_type)
    for each in range(num_of_landmarks):
        coordinates[each] = (original_landmarks.part(each).x, original_landmarks.part(each).y)

    return coordinates


# ==============================================================================
#   2.Draw regression line & find pupil function
# # Input: pictures & landmarks in numpy format
# # Output: Left pupil coordinate & right pupil coordinate
# ==============================================================================
def get_centers(image, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    right_eye_outer = landmarks[0]
    right_eye_inner = landmarks[1]
    left_eye_outer = landmarks[2]
    left_eye_inner = landmarks[3]

    coordinates = np.transpose(landmarks[0:4])
    x = coordinates[0]
    y = coordinates[1]

    A = np.transpose(np.vstack([x, np.ones(len(x))]))

    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (left_eye_inner[0] + left_eye_outer[0]) / 2
    x_right = (right_eye_inner[0] + right_eye_outer[0]) / 2

    eye_center_left = np.array([np.int32(x_left), np.int32(x_left * k + b)])
    eye_center_right = np.array([np.int32(x_right), np.int32(x_right * k + b)])


    # pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    # cv2.polylines(img, [pts], False, (255, 0, 0), 1)
    # cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    # cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return eye_center_left, eye_center_right


# ==============================================================================
#   3.Face alignment function
# # Input: picture & left pupil coordinates & right pupil coordinates
# # Output: Face image after alignment
# ==============================================================================
def get_aligned_face(image, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    centers = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)  # eyebrow
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx * dx + dy * dy)  # distance between eye pupils
    scale = desired_dist / dist  # scaling ratio
    angle = np.degrees(np.arctan2(dy, dx))  # rotation angle
    Rot = cv2.getRotationMatrix2D(centers, angle, scale)  # calculation of the rotation matrix

    # update the translation component of the matrix
    translation_X = desired_w * 0.5
    translation_Y = desired_h * 0.5
    Rot[0, 2] += (translation_X - centers[0])
    Rot[1, 2] += (translation_Y - centers[1])

    face = cv2.warpAffine(image, Rot, (desired_w, desired_h))

    return face


# ==============================================================================
# Whether to wear glasses discriminant function
# Input: Face picture after alignment
# Output: Discriminant value (True/False)
# ==============================================================================
def is_glasses_on(image):
    glasses_on = False
    image = cv2.GaussianBlur(image, (11, 11), 0)  # Gaussian Blur

    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=-1)  # y direction sobel edge detection
    sobel_y = cv2.convertScaleAbs(sobel_y)  # Convert back to uint8 type

    edges = sobel_y  # edge strength matrix

    # Otsu Binarization
    retVal, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate feature length

    d = len(thresh) * 0.5
    x = np.int32(d * 6 / 7)
    y = np.int32(d * 3 / 4)
    w = np.int32(d * 2 / 7)
    h = np.int32(d * 2 / 4)

    x_2_1 = np.int32(d * 1 / 4)
    x_2_2 = np.int32(d * 5 / 4)
    w_2 = np.int32(d * 1 / 2)
    y_2 = np.int32(d * 8 / 7)
    h_2 = np.int32(d * 1 / 2)

    roi_1 = thresh[y:y + h, x:x + w]  # Extract ROI
    roi_2_1 = thresh[y_2:y_2 + h_2, x_2_1:x_2_1 + w_2]
    roi_2_2 = thresh[y_2:y_2 + h_2, x_2_2:x_2_2 + w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure_1 = sum(sum(roi_1 / 255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])  # Calculate the evaluation value
    measure_2 = sum(sum(roi_2 / 255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])  # Calculate the evaluation value
    measure = measure_1 * 0.3 + measure_2 * 0.7

    # Determine the discriminant value according to the relationship between the evaluation value and the threshold
    if measure > 0.15:  # Threshold is adjustable, it is about 0.15 after test
        glasses_on = True
    else:
        glasses_on = False
    print(glasses_on)
    return glasses_on


# ==============================================================================
#   **************************Main function entry***********************************
# ==============================================================================

predictor_path = "./data/shape_predictor_5_face_landmarks.dat"  # Face key point training data path
detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor(predictor_path)  # Face keypoint detector -> predictor

cap = cv2.VideoCapture(0)  # Turn on the camera

while cap.isOpened():
    # Read video frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection
    rects = detector(gray, 1)
    # Face Detection
    for i, rect in enumerate(rects):
        # Get coordinates

        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face

        # Draw a border and add text annotations
        cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 0, 125), 2)
        # cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        #             (0, 255, 0), 2, cv2.LINE_AA)

        # Detect and mark landmarks
        landmarks = predictor(gray, rect)

        landmarks = landmarks_to_np(landmarks)
        # for (x, y) in landmarks:
        #     cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # Linear regression
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)

        # Face alignment
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)

        # Determine whether to wear glasses

        if is_glasses_on(aligned_face):
            glasses_boolean = True
        else:
            glasses_boolean = False

    # show result
    cv2.imshow("Result", img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:  # Press "Esc" to exit
        break

cap.release()
cv2.destroyAllWindows()
