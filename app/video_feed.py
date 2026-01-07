# app/video_feed.py

import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import time

# Internal module import
from app.pose_analyzer import analyze_pose

def process_and_display_frame(mentor_window):
    """
    Handles the frame update loop for the PoseMentor application.
    Reads frame, processes with MediaPipe, analyzes pose, logs data, 
    and updates the camera label.

    Args:
        mentor_window: The instance of the PoseMentor QMainWindow class.
    """ 
    # Access the instance's attributes directly
    cap = mentor_window.cap
    pose = mentor_window.pose
    mp_drawing = mentor_window.mp_drawing
    mp_pose = mentor_window.mp_pose
    
    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False # Read-only for efficiency (MediaPipe requirement)
    results = pose.process(rgb)
    rgb.flags.writeable = True

    # If landmarks found, draw and analyze
    if results.pose_landmarks:
        # Draw landmarks on the RGB copy
        mp_drawing.draw_landmarks(rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Analyze pose
        workout = mentor_window.workoutBox.currentText()
        feedback, color, pred_angle, correct_flag, ideal_angle = analyze_pose(results.pose_landmarks.landmark, workout)

        # Update feedback label
        mentor_window.feedbackLabel.setText(feedback)
        mentor_window.current_correct_flag = correct_flag
        
        # Speak feedback for audio guidance
        mentor_window.speak_feedback(feedback)
        
        # Color mapping
        if color.lower() == "green":
            style_color = "green"
        elif color.lower() == "red":
            style_color = "red"
        else:
            style_color = "orange"
        mentor_window.feedbackLabel.setStyleSheet(f"color: {style_color}; padding: 6px;")

        # Log metrics if ideal angle is defined
        if ideal_angle > 0:
            mentor_window.true_angles.append(ideal_angle)
            mentor_window.pred_angles.append(pred_angle)
            mentor_window.true_labels.append(1)  # ideal is always 'correct'
            mentor_window.pred_labels.append(correct_flag)

        # Update counters
        current_time = time.time() - mentor_window.start_timestamp if mentor_window.start_timestamp else 0
        hold_exercises = ["Plank", "Hollow Body Hold", "Superman Hold"]
        if workout in hold_exercises:
            mentor_window.hold_timer.update(correct_flag, current_time)
        else:
            mentor_window.rep_counter.update(pred_angle, correct_flag, workout)

        # Update metrics UI constantly
        mentor_window.calculate_metrics_ui(show_final=False)
        mentor_window.update_counter_display()

    # Convert RGB frame back to QImage for display
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg).scaled(mentor_window.cameraLabel.width(), mentor_window.cameraLabel.height(), Qt.KeepAspectRatio)
    mentor_window.cameraLabel.setPixmap(pix)