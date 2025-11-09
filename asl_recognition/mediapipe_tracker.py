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

        # ✅ initialize frame counter properly
        self.frame_count = 0

        # Different colors for left and right hands
        self.colors = {
            "Left": (255, 0, 0),   # Blue
            "Right": (0, 255, 0)   # Green
        }

    def extract_keypoints(self, image_bgr):
        """Extracts (x, y, z) coordinates for both hands and draws them."""
        image_bgr = cv2.flip(image_bgr, 1)  # selfie view
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        # ✅ increment frame counter
        self.frame_count += 1

        keypoints = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                color = self.colors.get(handedness, (0, 255, 255))

                self.mp_draw.draw_landmarks(
                    image_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints = [0.0] * (21 * 3 * 2)

        # ✅ draw frame counter
        cv2.putText(
            image_bgr,
            f"Frame: {self.frame_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return keypoints, image_bgr

    def reset_counter(self):
        """Reset the frame counter to zero."""
        self.frame_count = 0

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

        cv2.imshow("HandsUp - Mediapipe Tracker (Frame Counter)", flipped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total frames processed: {tracker.frame_count}")
