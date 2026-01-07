import os
# Must be set before importing PyQt5 on some systems
os.environ['QT_OPENGL'] = 'software' 
import sys
# PyQt5 UI Components and core modules
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QFrame
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QSize

# External dependencies for initialization
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3


# Internal module imports (delegation)
from app.metrics import RepCounter, HoldTimer
from app.metrics import calculate_metrics
from app.video_feed import process_and_display_frame # Used in update_frame


# ------------------------------
# PoseMentor Application Class
# ------------------------------
class PoseMentor(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window and icon
        self.setWindowTitle("PoseMentor -> AI-Powered Calisthenics Form Checker")
        try:
            # References the icon from the 'Icons' directory
            self.setWindowIcon(QIcon("Icons/posementor_icon.png"))
        except Exception:
            pass
        self.setGeometry(100, 100, 960, 840)

        # UI, camera, mediapipe
        self.initUI()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame) 

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Evaluation storage
        self.true_angles = []   # ground truth (ideal angle) per frame
        self.pred_angles = []   # predicted angle (from landmarks) per frame
        self.true_labels = []   # ground truth labels (1 = correct)
        self.pred_labels = []   # predicted labels (1 = app marks correct)

        # Rep and hold counters
        self.rep_counter = RepCounter()
        self.hold_timer = HoldTimer()
        self.start_timestamp = None
        self.current_correct_flag = 0

        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        self.last_feedback = ""  # Track last spoken feedback to avoid repetition

        # State
        self.running = False

    # ------------------------------
    # UI Construction
    # ------------------------------
    def initUI(self):
        # Top-left icon
        leftIcon = QLabel()
        try:
            pix = QPixmap("Icons/posementor_icon.png").scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            leftIcon.setPixmap(pix)
        except Exception:
            leftIcon.setText("🤸‍♂️")
            leftIcon.setFont(QFont("Arial", 24))

        # Title
        title = QLabel("PoseMentor")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        # Workout selector
        self.workoutBox = QComboBox()
        self.workoutBox.setIconSize(QSize(36, 36))
        workouts = [
            ("pushups_icon.png", "Pushups"),
            ("pullups_icon.png", "Pullups"),
            ("dips_icon.png", "Parallel Dips"),
            ("squats_icon.png", "Bodyweight Squats"),
            ("plank_icon.png", "Plank"),
            ("hollowbody_icon.png", "Hollow Body Hold"),
            ("superman_icon.png", "Superman Hold"),
            ("legraises_icon.png", "Hanging Leg Raises"),
        ]
        for icon, name in workouts:
            icon_path = f"Icons/{icon}"
            if os.path.exists(icon_path):
                self.workoutBox.addItem(QIcon(icon_path), name)
            else:
                self.workoutBox.addItem(name)

        # Start and Stop buttons
        self.startButton = QPushButton("Start")
        try:
            self.startButton.setIcon(QIcon("Icons/start_icon.png"))
            self.startButton.setIconSize(QSize(24, 24))
        except Exception:
            pass
        self.startButton.clicked.connect(self.start_workout)

        self.stopButton = QPushButton("Stop")
        try:
            self.stopButton.setIcon(QIcon("Icons/stop_icon.png"))
            self.stopButton.setIconSize(QSize(24, 24))
        except Exception:
            pass
        self.stopButton.clicked.connect(self.stop_workout)
        self.stopButton.setEnabled(False)

        # Top panel layout
        top_panel = QHBoxLayout()
        top_panel.addWidget(leftIcon)
        top_panel.addWidget(title)
        top_panel.addStretch(1)
        top_panel.addWidget(QLabel("Select Workout:"))
        top_panel.addWidget(self.workoutBox)
        top_panel.addWidget(self.startButton)
        top_panel.addWidget(self.stopButton)

        # Camera preview
        self.cameraLabel = QLabel()
        self.cameraLabel.setFixedSize(780, 640)
        self.cameraLabel.setStyleSheet("background-color: black; border: 2px solid #444;")
        self.cameraLabel.setAlignment(Qt.AlignCenter)

        # Feedback label
        self.feedbackLabel = QLabel("Select workout and press Start")
        self.feedbackLabel.setFont(QFont("Arial", 13))
        self.feedbackLabel.setAlignment(Qt.AlignCenter)
        self.feedbackLabel.setStyleSheet("color: green; padding: 6px;")

        # Metrics area: labels (black) + values (orange)
        metrics_frame = QFrame()
        metrics_layout = QHBoxLayout()
        metrics_frame.setLayout(metrics_layout)
        metrics_frame.setFrameShape(QFrame.StyledPanel)

        # MAE
        mae_label = QLabel("MAE:")
        mae_label.setFont(QFont("Arial", 11))
        mae_label.setStyleSheet("color: black;")
        self.mae_value = QLabel("-")
        self.mae_value.setFont(QFont("Arial", 11, QFont.Bold))
        self.mae_value.setStyleSheet("color: orange;")
        mae_box = QVBoxLayout()
        mae_box.addWidget(mae_label, alignment=Qt.AlignCenter)
        mae_box.addWidget(self.mae_value, alignment=Qt.AlignCenter)

        # MSE
        mse_label = QLabel("MSE:")
        mse_label.setFont(QFont("Arial", 11))
        mse_label.setStyleSheet("color: black;")
        self.mse_value = QLabel("-")
        self.mse_value.setFont(QFont("Arial", 11, QFont.Bold))
        self.mse_value.setStyleSheet("color: orange;")
        mse_box = QVBoxLayout()
        mse_box.addWidget(mse_label, alignment=Qt.AlignCenter)
        mse_box.addWidget(self.mse_value, alignment=Qt.AlignCenter)

        # F1
        f1_label = QLabel("F1:")
        f1_label.setFont(QFont("Arial", 11))
        f1_label.setStyleSheet("color: black;")
        self.f1_value = QLabel("-")
        self.f1_value.setFont(QFont("Arial", 11, QFont.Bold))
        self.f1_value.setStyleSheet("color: orange;")
        f1_box = QVBoxLayout()
        f1_box.addWidget(f1_label, alignment=Qt.AlignCenter)
        f1_box.addWidget(self.f1_value, alignment=Qt.AlignCenter)

        # Add metric boxes to metrics_layout
        metrics_layout.addLayout(mae_box)
        metrics_layout.addSpacing(20)
        metrics_layout.addLayout(mse_box)
        metrics_layout.addSpacing(20)
        metrics_layout.addLayout(f1_box)

        # Counter label for reps or time
        self.counter_label = QLabel("Reps: 0")
        self.counter_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.counter_label.setStyleSheet("color: blue; padding: 6px;")
        self.counter_label.setAlignment(Qt.AlignCenter)

        # Bottom layout combining feedback, metrics, and counter
        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.feedbackLabel)
        
        # Horizontal layout for metrics and counter
        metrics_and_counter = QHBoxLayout()
        metrics_and_counter.addWidget(metrics_frame)
        metrics_and_counter.addStretch(1)
        metrics_and_counter.addWidget(self.counter_label)
        
        bottom_layout.addLayout(metrics_and_counter)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_panel)
        main_layout.addSpacing(10)

        # Center with camera and bottom
        center_h = QHBoxLayout()
        center_h.addStretch(1)
        center_h.addWidget(self.cameraLabel)
        center_h.addStretch(1)

        main_layout.addLayout(center_h)
        main_layout.addSpacing(10)
        main_layout.addLayout(bottom_layout)

        # Container widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ------------------------------
    # Workout Control Methods
    # ------------------------------
    def start_workout(self):
        if self.running:
            return
        self.running = True
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.workoutBox.setEnabled(False)
        self.feedbackLabel.setText(f"Started: {self.workoutBox.currentText()} (Perform exercise)")
        self.feedbackLabel.setStyleSheet("color: green; padding: 6px;")
        # Clear previous metrics storage
        self.true_angles.clear()
        self.pred_angles.clear()
        self.true_labels.clear()
        self.pred_labels.clear()
        # Reset counters
        self.rep_counter.reset()
        self.hold_timer.reset()
        self.start_timestamp = time.time()
        self.update_counter_display()
        # Start camera
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        
        # Voice announcement
        self.speak_feedback(f"Starting {self.workoutBox.currentText()}. Get ready to begin.")

    def stop_workout(self):
        if not self.running:
            return
        self.running = False
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.workoutBox.setEnabled(True)
        self.timer.stop()
        # release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        # Compute final metrics and show summary
        self.calculate_and_show_metrics(final=True)
        self.feedbackLabel.setText("Workout stopped. Metrics updated.")
        self.feedbackLabel.setStyleSheet("color: blue; padding: 6px;")
        
        # Voice announcement with final results
        workout = self.workoutBox.currentText()
        hold_exercises = ["Plank", "Hollow Body Hold", "Superman Hold"]
        if workout in hold_exercises:
            performance = f"held for {self.hold_timer.total_hold_time:.1f} seconds"
        else:
            performance = f"completed {self.rep_counter.rep_count} reps"
        
        self.speak_feedback(f"Workout completed. You {performance} in {workout}.")

    # ------------------------------
    # Metrics Calculation & Display
    # ------------------------------
    def calculate_metrics_ui(self, show_final=False):
        #Calculates metrics by calling app.metrics.calculate_metrics and updates UI labels.
        
        # Note: calculate_metrics handles empty data check internally
        mae, mse, f1 = calculate_metrics(self.true_angles, self.pred_angles, self.true_labels, self.pred_labels)

        # Update UI (values in orange)
        self.mae_value.setText(f"{mae:.2f}" if not np.isnan(mae) else "-")
        self.mse_value.setText(f"{mse:.2f}" if not np.isnan(mse) else "-")
        self.f1_value.setText(f"{f1:.2f}")

        if show_final:
            # Show a summary message box
            workout = self.workoutBox.currentText()
            hold_exercises = ["Plank", "Hollow Body Hold", "Superman Hold"]
            if workout in hold_exercises:
                performance = f"Hold Time: {self.hold_timer.total_hold_time:.1f} seconds"
            else:
                performance = f"Reps Completed: {self.rep_counter.rep_count}"
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Session Metrics")
            msg.setText(
                f"Final Metrics for {workout}:\n\n"
                f"{performance}\n\n"
                f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nF1 Score: {f1:.2f}\n\n"
                "(Angles measured in degrees; labels: 1=correct, 0=incorrect)"
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def calculate_and_show_metrics(self, final=False):
        # wrapper for clarity
        self.calculate_metrics_ui(show_final=final)

    def update_counter_display(self):
        workout = self.workoutBox.currentText()
        hold_exercises = ["Plank", "Hollow Body Hold", "Superman Hold"]
        if workout in hold_exercises:
            current_time = time.time() - self.start_timestamp if self.start_timestamp else 0
            hold_time = self.hold_timer.update(self.current_correct_flag, current_time)
            self.counter_label.setText(f"Hold: {hold_time:.1f}s")
            
            # Voice feedback for hold milestones
            if int(hold_time) % 10 == 0 and int(hold_time) > 0 and not hasattr(self, 'last_hold_announcement') or getattr(self, 'last_hold_announcement', 0) != int(hold_time):
                self.last_hold_announcement = int(hold_time)
                self.speak_feedback(f"Great job! You've held for {int(hold_time)} seconds.")
        else:
            current_reps = self.rep_counter.rep_count
            self.counter_label.setText(f"Reps: {current_reps}")
            
            # Voice feedback for rep milestones
            if current_reps > 0 and current_reps % 5 == 0 and not hasattr(self, 'last_rep_announcement') or getattr(self, 'last_rep_announcement', 0) != current_reps:
                self.last_rep_announcement = current_reps
                self.speak_feedback(f"Excellent! You've completed {current_reps} reps.")

    def speak_feedback(self, feedback_text):
        """Speak feedback text using TTS, but avoid repeating the same feedback too frequently"""
        if feedback_text != self.last_feedback:
            self.last_feedback = feedback_text
            try:
                self.tts_engine.say(feedback_text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")  # Silent fail if TTS has issues

    # ------------------------------
    # Frame Update Loop (Delegated to video_feed.py)
    # ------------------------------
    def update_frame(self):
        # Delegate frame processing to the external function/module, passing 'self'
        process_and_display_frame(self)

    # ------------------------------
    # Close Event
    # ------------------------------
    def closeEvent(self, event):
        # Stop timer and release camera
        try:
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        # Show final metrics if session had data
        if len(self.true_angles) > 0:
            self.calculate_and_show_metrics(final=True)
        event.accept() 
