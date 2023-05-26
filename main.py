import math

import cv2
import mediapipe as mp

import numpy
import pyautogui

import autopy
from mouse import move, wheel
import time

from stopwatch import Stopwatch

# Mediapipe decleration
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# screen decleration for moving mouse
screenWidth, screenHeight = autopy.screen.size()
px, py = 0, 0
cx, cy = 0, 0
# Set up video capture
cap = cv2.VideoCapture(0)


# right wrist variables
right_wrist_array_x = []
right_wrist_array_y = []
right_wrist_average_array_x = []
right_wrist_average_array_y = []
right_wrist_stopwatch_horizontal_flag = True
right_wrist_stopwatch_horizontal = Stopwatch()
right_wrist_stopwatch_vertical_flag = True
right_wrist_stopwatch_vertical = Stopwatch()
right_wrist_frame_counter = 0

#left wrist variables
left_wrist_array_x = []
left_wrist_array_y = []
left_wrist_average_array_x = []
left_wrist_average_array_y = []
left_wrist_stopwatch_horizontal_flag = True
left_wrist_stopwatch_horizontal = Stopwatch()
left_wrist_stopwatch_vertical_flag = True
left_wrist_stopwatch_vertical = Stopwatch()
left_wrist_frame_counter = 0


#Toggles mouse movement with nose
moveMouseFlag=True


def calculate_angle(landmark1, landmark2, landmark3):
    if (
            landmark1.visibility > 0.7
            and landmark2.visibility > 0.7
            and landmark3.visibility > 0.7
    ):
        angle = math.degrees(
            math.atan2(landmark3.y - landmark2.y, landmark3.x - landmark2.x)
            - math.atan2(landmark1.y - landmark2.y, landmark1.x - landmark2.x)
        )

    return angle


# Initiate pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        # Read a frame from the video stream
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect pose landmarks
        results = pose.process(image)

        # Convert the image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmark positions
        if results.pose_landmarks:
            # Get the position of the nose landmark
            nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            # Right Shoulder
            right_shoulder = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]
            # Right Elbow
            right_elbow = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_ELBOW
            ]
            # Left Shoulder
            left_shoulder = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ]
            left_shoulder_visibility = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ].visibility
            # Left Elbow
            left_elbow = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ELBOW
            ]
            # left wrist
            left_wrist = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_WRIST
            ]
            # right wrist
            right_wrist = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
            # left hip
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            # left knee
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            # left ankle
            left_ankle = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ]
            # right hip
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            # left knee
            right_knee = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_KNEE
            ]
            # left ankle
            right_ankle = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]

            # Move mouse with nose
            if nose_landmark.visibility > 0.7:
                x3 = numpy.interp((nose_landmark.x * image.shape[1]), (0, 640), (0, screenWidth))
                y3 = numpy.interp((nose_landmark.y * image.shape[0]), (0, 480), (0, screenHeight))
                cx = px + (x3 - px) / 9
                cy = py + (y3 - py) / 9
                # Move the mouse cursor
                if moveMouseFlag:
                    print(" Mouse moving with nose")
                    autopy.mouse.move(screenWidth-cx, cy)
                px, py = cx, cy

            if right_wrist.visibility > 0.7:

                right_wrist_frame_counter += 1
                right_wrist_array_x.append(right_wrist.x)
                right_wrist_array_y.append(right_wrist.y)

                if right_wrist_frame_counter % 15 == 0:
                    right_wrist_average_array_x.append(sum(right_wrist_array_x) / len(right_wrist_array_x))
                    right_wrist_average_array_y.append(sum(right_wrist_array_y) / len(right_wrist_array_y))
                    right_wrist_array_x.clear()
                    right_wrist_array_y.clear()

                    if len(right_wrist_average_array_x) >= 2:
                        x, y = right_wrist_average_array_x[-2:]
                        # Swipe left with right hand (left click )
                        if y - x >= 0.15 and (
                                right_wrist_stopwatch_horizontal.duration > 1.5 or right_wrist_stopwatch_horizontal_flag):
                            right_wrist_stopwatch_horizontal.reset()
                            right_wrist_stopwatch_horizontal.start()
                            print("Swipe Left detected")
                            pyautogui.click(button='left')
                            right_wrist_stopwatch_horizontal_flag = False
                            right_wrist_average_array_x.clear()
                        # Swipe right with right hand (right click )
                        elif y - x <= -0.15 and (
                                right_wrist_stopwatch_horizontal.duration > 1.5 or right_wrist_stopwatch_horizontal_flag):
                            right_wrist_stopwatch_horizontal.reset()
                            right_wrist_stopwatch_horizontal.start()
                            pyautogui.click(button='right')
                            print("Swipe Right detected")
                            right_wrist_stopwatch_horizontal_flag = False
                            right_wrist_average_array_x.clear()

                    if len(right_wrist_average_array_y) >= 2:
                        x, y = right_wrist_average_array_y[-2:]
                        # Swipe up right hand (Scroll down)
                        if y - x >= 0.2 and (
                                right_wrist_stopwatch_vertical.duration > 1.5 or right_wrist_stopwatch_vertical_flag):
                            right_wrist_stopwatch_vertical.reset()
                            right_wrist_stopwatch_vertical.start()
                            print("scrolling down")
                            wheel(-1)
                            right_wrist_stopwatch_vertical_flag = False
                            right_wrist_average_array_y.clear()
                        #Swipe up right hand (Scroll up)
                        elif y - x <= -0.2 and (
                                right_wrist_stopwatch_vertical.duration > 1.5 or right_wrist_stopwatch_vertical_flag):
                            right_wrist_stopwatch_vertical.reset()
                            right_wrist_stopwatch_vertical.start()
                            print("scrolling up")
                            wheel(1)
                            right_wrist_stopwatch_vertical_flag = False
                            right_wrist_average_array_y.clear()


            #Out of frame right hand
            else:
                right_wrist_frame_counter = 0
                right_wrist_array_x.clear()
                right_wrist_array_y.clear()
                right_wrist_average_array_x.clear()
                right_wrist_average_array_y.clear()
                right_wrist_stopwatch_horizontal.reset()
                right_wrist_stopwatch_vertical.reset()
                right_wrist_stopwatch_horizontal_flag = True
                right_wrist_stopwatch_vertical_flag = True

            if left_wrist.visibility > 0.7:

                left_wrist_frame_counter += 1
                left_wrist_array_x.append(left_wrist.x)
                left_wrist_array_y.append(left_wrist.y)

                if left_wrist_frame_counter % 15 == 0:
                    left_wrist_average_array_x.append(sum(left_wrist_array_x) / len(left_wrist_array_x))
                    left_wrist_average_array_y.append(sum(left_wrist_array_y) / len(left_wrist_array_y))
                    left_wrist_array_x.clear()
                    left_wrist_array_y.clear()

                    if len(left_wrist_average_array_x) >= 2:
                        x, y = left_wrist_average_array_x[-2:]

                        #Swipe Left with left hand (Double click)
                        if y - x >= 0.15 and (
                                left_wrist_stopwatch_horizontal.duration > 1.5 or left_wrist_stopwatch_horizontal_flag):
                            left_wrist_stopwatch_horizontal.reset()
                            left_wrist_stopwatch_horizontal.start()
                            print("Double click")
                            pyautogui.doubleClick()
                            left_wrist_stopwatch_horizontal_flag = False
                            left_wrist_average_array_x.clear()
                        # Swipe right with left hand
                        elif y - x <= -0.15 and (
                                left_wrist_stopwatch_horizontal.duration > 1.5 or left_wrist_stopwatch_horizontal_flag):
                            left_wrist_stopwatch_horizontal.reset()
                            left_wrist_stopwatch_horizontal.start()
                            print(" Mouse not moving with nose")
                            moveMouseFlag=not moveMouseFlag
                            left_wrist_stopwatch_horizontal_flag = False
                            left_wrist_average_array_x.clear()


                    if len(left_wrist_average_array_y) >= 2:
                        x, y = left_wrist_average_array_y[-2:]

                        # Swipe Up with left hand (Drop)
                        if y - x >= 0.2 and (
                                left_wrist_stopwatch_vertical.duration > 1.5 or left_wrist_stopwatch_vertical_flag):
                            left_wrist_stopwatch_vertical.reset()
                            left_wrist_stopwatch_vertical.start()
                            print("Drag")
                            pyautogui.mouseDown();
                            left_wrist_stopwatch_vertical_flag = False
                            left_wrist_average_array_y.clear()
                        # Swipe Down with left hand (Drag)
                        elif y - x <= -0.2 and (
                                left_wrist_stopwatch_vertical.duration > 1.5 or left_wrist_stopwatch_vertical_flag):
                            left_wrist_stopwatch_vertical.reset()
                            left_wrist_stopwatch_vertical.start()
                            print("Drop")
                            pyautogui.mouseUp()
                            left_wrist_stopwatch_vertical_flag = False
                            left_wrist_average_array_y.clear()


            #Out of frame left hand
            else:
                left_wrist_frame_counter = 0
                left_wrist_array_x.clear()
                left_wrist_array_y.clear()
                left_wrist_average_array_x.clear()
                left_wrist_average_array_y.clear()
                left_wrist_stopwatch_horizontal.reset()
                left_wrist_stopwatch_vertical.reset()
                left_wrist_stopwatch_horizontal_flag = True
                left_wrist_stopwatch_vertical_flag = True


        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Show the annotated image

        cv2.imshow("Pose Tracking", cv2.flip(image, 1))
        #cv2.moveWindow('Pose Tracking', 0, 0)
        #cv2.setWindowProperty('Pose Tracking', cv2.WND_PROP_TOPMOST, 1)  # Set the window as always on top

        # Exit on ESC key press
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
