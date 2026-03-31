import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup paths
model_path = 'pose_landmarker.task' # Ensure this file is in your folder!
video_input = 'videoplayback.mp4'   
video_output = 'analyzed_match.mp4'

# 2. Configure MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# 3. Open Video
cap = cv2.VideoCapture(video_input)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Video Writer to save the analysis
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Convert for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Detect
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if results.pose_landmarks:
            for landmarks in results.pose_landmarks:
                # Right Shoulder(12), Elbow(14), Wrist(16)
                s = [landmarks[12].x, landmarks[12].y]
                e = [landmarks[14].x, landmarks[14].y]
                w = [landmarks[16].x, landmarks[16].y]

                angle = calculate_angle(s, e, w)
                
                # Shot Classification Logic
                label = "Positioning..."
                color = (255, 255, 255)
                
                if w[1] > 0.75: # Low wrist
                    label = "ANALYSIS: DINK"
                    color = (0, 255, 0)
                elif w[1] < 0.35 and angle > 155: # High & Straight
                    label = "ANALYSIS: SMASH"
                    color = (0, 0, 255)

                # Visual Feedback
                cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
                cv2.putText(frame, f"Elbow Angle: {int(angle)}deg", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Write and Show
        out.write(frame)
        cv2.imshow('Coach Analysis Mode', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Analysis complete! Saved as {video_output}")