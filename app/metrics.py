import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

class RepCounter:
    def __init__(self):
        self.rep_count = 0
        self.phase = "up"  # "up", "down", "bottom"
        self.last_angle = None

    def update(self, pred_angle, correct_flag, workout):
        if workout == "Pushups":
            if pred_angle > 160 and self.phase != "up":
                if self.phase == "bottom":
                    self.rep_count += 1
                self.phase = "up"
            elif pred_angle < 90 and correct_flag == 1 and self.phase == "up":
                self.phase = "down"
            elif pred_angle >= 90 and pred_angle <= 160 and self.phase == "down":
                self.phase = "bottom"
        elif workout in ["Pullups", "Parallel Dips"]:
            # Similar logic for pullups and dips
            if pred_angle > 150 and self.phase != "down":
                if self.phase == "up":
                    self.rep_count += 1
                self.phase = "down"
            elif pred_angle < 90 and correct_flag == 1 and self.phase == "down":
                self.phase = "up"
        elif workout == "Bodyweight Squats":
            if pred_angle > 150 and self.phase != "up":
                if self.phase == "down":
                    self.rep_count += 1
                self.phase = "up"
            elif pred_angle < 90 and correct_flag == 1 and self.phase == "up":
                self.phase = "down"
        elif workout == "Hanging Leg Raises":
            if pred_angle > 100 and self.phase != "down":
                if self.phase == "up":
                    self.rep_count += 1
                self.phase = "down"
            elif pred_angle < 80 and correct_flag == 1 and self.phase == "down":
                self.phase = "up"
        
        self.last_angle = pred_angle
        return self.rep_count

    def reset(self):
        self.rep_count = 0
        self.phase = "up"
        self.last_angle = None

class HoldTimer:
    def __init__(self):
        self.start_time = None
        self.total_hold_time = 0.0
        self.current_hold_start = None
        self.is_holding = False

    def update(self, correct_flag, timestamp):
        if correct_flag == 1:
            if not self.is_holding:
                self.current_hold_start = timestamp
                self.is_holding = True
        else:
            if self.is_holding:
                self.total_hold_time += timestamp - self.current_hold_start
                self.is_holding = False
        return self.total_hold_time + (timestamp - self.current_hold_start if self.is_holding else 0)

    def reset(self):
        self.start_time = None
        self.total_hold_time = 0.0
        self.current_hold_start = None
        self.is_holding = False

def calculate_metrics(true_angles, pred_angles, true_labels, pred_labels):
    if len(true_angles) < 2:
        return float('nan'), float('nan'), 0.0

    y_true_angles = np.array(true_angles)
    y_pred_angles = np.array(pred_angles)
    y_true_labels = np.array(true_labels)
    y_pred_labels = np.array(pred_labels)

    try:
        # Regression Metrics (for angle error)
        mae = mean_absolute_error(y_true_angles, y_pred_angles)
        mse = mean_squared_error(y_true_angles, y_pred_angles)

        # Classification Metric (for correct/incorrect label agreement)
        f1 = f1_score(y_true_labels, y_pred_labels, zero_division=1)
    except Exception:
        mae = float('nan')
        mse = float('nan')
        f1 = 0.0

    return mae, mse, f1 
