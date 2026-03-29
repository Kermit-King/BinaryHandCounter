import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'hand_landmarker.task' 

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

tip_ids = [8, 12, 16, 20]

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        binary_val = [0, 0, 0, 0, 0]

        if result.hand_landmarks:
            
            hand_lms = result.hand_landmarks[0]

            if hand_lms[17].x > hand_lms[3].x:
                if hand_lms[4].x < hand_lms[3].x:
                    binary_val[0] = 1
            else:
                if hand_lms[4].x > hand_lms[3].x:
                    binary_val[0] = 1

            for i in range(0, 4):
                if hand_lms[tip_ids[i]].y < hand_lms[tip_ids[i]-2].y:
                    binary_val[i+1] = 1

            decimal_out = sum(val * (2**i) for i, val in enumerate(binary_val))

            cv2.putText(frame, f'Binary: {binary_val}', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame, f'Decimal: {decimal_out}', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Modern Hand Binary Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()