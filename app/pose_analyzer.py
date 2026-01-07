import math
# ------------------------------
# Math Helpers
# ------------------------------
def get_angle(a, b, c):
    # Calculate angle using atan2 for robustness
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) -
                        math.atan2(a[1] - b[1], a[0] - b[0]))
    
    ang = abs(ang)
    if ang > 360:
        ang = ang % 360
    # Convert to smallest angle (max 180)
    if ang > 180:
        ang = 360 - ang
    return ang

# ------------------------------
# Pose Analysis Logic
# ------------------------------
def analyze_pose(landmarks, workout):
    # default
    feedback = "Good form! Keep it up."
    color = "green"
    correct_flag = 1
    ideal_angle = 0
    pred_angle = 0.0

    # Helper function to get coordinates
    def lm_coords(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    # Helper function to get angle using the imported get_angle
    a = lambda i, j, k: get_angle(lm_coords(i), lm_coords(j), lm_coords(k))

    # MediaPipe Indices used: 
    # 11: L_Shoulder, 12: R_Shoulder, 13: L_Elbow, 14: R_Elbow
    # 15: L_Wrist, 16: R_Wrist, 23: L_Hip, 24: R_Hip
    # 25: L_Knee, 26: R_Knee, 27: L_Ankle, 28: R_Ankle

    try:
        if workout == "Pushups":
            # Elbow angle (low point around 90)
            pred_angle = a(12, 14, 16)  # right elbow (shoulder-elbow-wrist)
            ideal_angle = 90
            if pred_angle > 160:
                feedback = "Not low enough. Lower your chest."
                color = "red"
                correct_flag = 0
            elif pred_angle < 40:
                feedback = "Too deep. Maintain neutral spine."
                color = "red"
                correct_flag = 0

        elif workout == "Pullups":
            # Elbow angle (fully retracted around 60)
            pred_angle = a(12, 14, 16)
            ideal_angle = 60
            if pred_angle > 150:
                feedback = "Pull higher. Elbows too extended."
                color = "red"
                correct_flag = 0
            elif pred_angle < 40:
                feedback = "Too high - control descent."
                color = "red"
                correct_flag = 0

        elif workout == "Parallel Dips":
            # Elbow angle (bottom position around 75-90)
            pred_angle = a(12, 14, 16)
            ideal_angle = 75
            if pred_angle > 150:
                feedback = "Not dipping deep enough."
                color = "red"
                correct_flag = 0
            elif pred_angle < 50:
                feedback = "Too low. Risk of shoulder strain."
                color = "red"
                correct_flag = 0

        elif workout == "Bodyweight Squats":
            # Knee angle (parallel squat around 90)
            pred_angle = a(24, 26, 28)  # right knee (hip-knee-ankle)
            ideal_angle = 90
            if pred_angle > 150:
                feedback = "Not squatting deep enough."
                color = "red"
                correct_flag = 0
            elif pred_angle < 60:
                feedback = "Too deep. Avoid knee strain."
                color = "red"
                correct_flag = 0

        elif workout == "Plank":
            # Torso/leg alignment (straight line near 180)
            pred_angle = a(12, 24, 28)  # right shoulder-right hip-right ankle
            ideal_angle = 180
            if pred_angle < 160:
                feedback = "Body not straight. Tighten your core."
                color = "red"
                correct_flag = 0

        elif workout == "Hollow Body Hold":
            # Overall body curvature/bend (approximation of hip-shoulder bend)
            pred_angle = a(12, 24, 26)  # shoulder-hip-knee angle approx
            ideal_angle = 100
            if pred_angle > 120:
                feedback = "Loosen core; lift shoulders and legs more."
                color = "red"
                correct_flag = 0

        elif workout == "Superman Hold":
            # Back extension/lift (approximation of straight line from back)
            pred_angle = a(11, 23, 27)  # left shoulder-left hip-left ankle
            ideal_angle = 160
            if pred_angle < 140:
                feedback = "Lift chest and legs higher."
                color = "red"
                correct_flag = 0

        elif workout == "Hanging Leg Raises":
            # Knee bend or raise height (straight legs/hip bend near 70-90)
            pred_angle = a(24, 26, 28)  # right hip-right knee-right ankle (if bent-leg raise)
            ideal_angle = 70
            if pred_angle < 70:
                feedback = "Raise legs higher."
                color = "red"
                correct_flag = 0

        else:
            # fallback
            pred_angle = a(12, 14, 16)
            ideal_angle = 90

    except Exception:
        # if landmarks missing or out of range
        feedback = "Pose not fully visible."
        color = "orange"
        correct_flag = 0
        pred_angle = 0
        ideal_angle = 0

    return feedback, color, pred_angle, correct_flag, ideal_angle