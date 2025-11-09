👐 HandsUp — Real-Time ASL to Speech Translator

🧠 Built at SBUHacks 2025

HandsUp bridges the communication gap between Deaf and non-signing individuals by translating American Sign Language (ASL) gestures into spoken English in real time.

Using MediaPipe, TensorFlow, and ElevenLabs, HandsUp captures live webcam input, classifies ASL gestures, and speaks them out loud using AI-generated voice.

⸻

🚀 Features

✅ Real-time Hand Tracking – Uses Google’s MediaPipe to detect and track both hands continuously.
✅ Gesture Recognition – A trained TensorFlow model classifies sequences of hand keypoints into ASL phrases.
✅ Speech Output – ElevenLabs Text-to-Speech converts recognized phrases into natural-sounding speech.
✅ Accessible Design – Built to help non-signers understand signed communication instantly.

⸻

🧩 Current Demo Scope (MVP)

This prototype recognizes 5 core ASL phrases:
1. Hello
2. Yes
3. Thank You
4. I'm Happy
5. See you soon 

💡 Future versions can be easily expanded — just record and train additional gesture sequences.

Setup Instructions

1.  Clone the repository
git clone https://github.com/hets-10/Hands-Up.git
cd Hands-Up

2. Create and Activate a Virtual Enviornment
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt


Running the Project

You must run the project from the repository root 
Run command:
python -m asl_recognition.predict_live

What happens:
	1.	Webcam feed opens.
	2.	Hands are tracked in real time.
	3.	Recognized gestures appear on screen with confidence %.
	4.	The system speaks the corresponding phrase aloud via ElevenLabs.
	5.	Press q to quit the demo.

🧠 Technical Highlights
	•	Built using TensorFlow Sequential model trained on frame-based keypoint sequences
	•	Uses MediaPipe Hands for landmark extraction
	•	Smooth real-time performance using a deque buffer for 30-frame sequences
	•	Dynamic overlay (bottom text box) for gesture info and debug stats
	•	Scalable design: easily add new phrases with minimal retraining

⸻

💡 Future Enhancements
	•	Expand dataset for 100+ ASL phrases
	•	Add bi-directional translation (English ↔ ASL)
	•	Integrate directly into Zoom / Meet / FaceTime via API overlay
	•	Deploy as a web app with live streaming inference
	•	Add subtitles and emotion detection for richer conversation context

Team
1. Het Shah
2. Rumman Khan
3. Alejandro Morales
4. Ibrahim Quaizar

🏁 Summary

HandsUp demonstrates that accessible AI communication tools can be built quickly and meaningfully.
Even with 5 phrases, this MVP proves real-time ASL-to-speech translation is possible — paving the way for more inclusive conversations everywhere.

“HandsUp — Because communication shouldn’t have barriers.” 🫶