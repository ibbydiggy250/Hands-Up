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
        self.frame_count = 0

        # Different colors for left and right hands
        self.colors = {
            "Left": (255, 0, 0),   # Blue
            "Right": (0, 255, 0)   # Green
        }

    def extract_keypoints(self, image_bgr):
        """Extracts keypoints for both hands safely and keeps stream alive."""
        # Flip for selfie view
        image_bgr = cv2.flip(image_bgr, 1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        self.frame_count += 1

        keypoints = []

        # Draw detected hands and collect keypoints
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                color = self.colors.get(handedness, (0, 255, 255))

                # Draw the hand
                self.mp_draw.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

                # Save coordinates for this hand
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

            # ✅ Pad missing hand (ensure shape consistency)
            num_hands = len(results.multi_hand_landmarks)
            if num_hands == 1:
                keypoints.extend([0.0] * (21 * 3))  # pad one missing hand
        else:
            # ✅ No hands detected — pad both
            keypoints = [0.0] * (21 * 3 * 2)

        # ================== Overlay Info (Bottom) ==================
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        h, w, _ = image_bgr.shape
        overlay = image_bgr.copy()

        # Create semi-transparent rectangle at bottom
        rect_height = 100
        cv2.rectangle(overlay, (0, h - rect_height), (w, h), (0, 0, 0), -1)
        image_bgr = cv2.addWeighted(overlay, 0.4, image_bgr, 0.6, 0)

        # Text lines
        lines = [
            f"Hands detected: {num_hands}",
            f"Frame: {self.frame_count}",
            "Press 'q' to Quit"
        ]

        # Draw each line with spacing and outline near bottom
        y0 = h - rect_height + 35
        dy = 30
        for i, text in enumerate(lines):
            y = y0 + i * dy
            # Shadow (black outline)
            cv2.putText(image_bgr, text, (12, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            # Foreground (colored text)
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            cv2.putText(image_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)

        return keypoints, image_bgr

    def reset_counter(self):
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

        keypoints, output = tracker.extract_keypoints(frame)
        cv2.imshow("HandsUp - Mediapipe Tracker (Stable)", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total frames processed: {tracker.frame_count}")