import cv2
import mediapipe as mp

class MediapipeTracker:
    def __init__(self, max_num_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Different colors for left and right hands
        self.colors = {
            "Left": (255, 0, 0),   # Blue
            "Right": (0, 255, 0)   # Green
        }

    def extract_keypoints(self, image_bgr):
        """
        Extracts (x, y, z) coordinates for both hands and draws them color-coded.
        Returns:
            keypoints (list of float): flattened list of hand landmark coordinates.
            flipped_frame (ndarray): horizontally flipped frame for display.
        """
        # Flip for natural selfie view
        image_bgr = cv2.flip(image_bgr, 1)

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        keypoints = []
        h, w, _ = image_bgr.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label  # "Left" or "Right"
                color = self.colors.get(handedness, (0, 255, 255))

                # Draw landmarks on the flipped image
                self.mp_draw.draw_landmarks(
                    image_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

                # Collect keypoints
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

        else:
            # No hands detected — fill with zeros for both hands
            keypoints = [0.0] * (21 * 3 * 2)

        return keypoints, image_bgr

    def close(self):
        self.hands.close()


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = MediapipeTracker()

    print("[INFO] Starting hand tracking... Press 'q' to quit.")
    while True:
        success, frame = cap.read()
        if not success:
            break

        _, flipped_frame = tracker.extract_keypoints(frame)

        cv2.putText(flipped_frame, "Press 'q' to Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("HandsUp - Mediapipe Tracker (Flipped)", flipped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Hand tracking stopped.")
