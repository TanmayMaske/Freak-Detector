# Freak Detector — Real-Time Gesture → GIF System

Freak Detector is a real-time computer-vision project that detects face and hand gestures using **MediaPipe Holistic** and plays corresponding **GIF animations** on-screen.
This project is built in Python using **OpenCV**, **MediaPipe**, **NumPy**, and **ImageIO**.

#🧠 Features

* **Quick Nod → Blink** gesture detection
* **Hand Mesh recognition** (improves accuracy of hands_up, tuff, rubbing, idea)
* **Tongue / Mouth-open detection**
* **Finger-in-mouth → Thinking**
* **Joined hands → Rubbing hands**
* **Weighing hands → Tuff gesture**
* **Stable GIF transitions**
* **Full GIF playback per gesture (no skipping)**
* **Smooth output panel combining camera + GIF**

## 📂 Project Structure

```
FreakDetector/
│
├── freak_detector.py
├── assets/
│      ├── tuff.gif
│      ├── thinking.gif
│      ├── idea.gif
│      ├── blink.gif
│      ├── tongue.gif
│      ├── hands_up.gif
│      ├── rubbing_hands.gif
│
└── README.md
```

## 📦 Requirements

Make sure you have **Python 3.8+** installed.

Install required libraries:

```bash
pip install opencv-python mediapipe imageio numpy
```

---

## ▶️ How to Run

Open terminal in the project folder:

```bash
python freak_detector.py
```

Press **Q** to quit.

---

## ✋ Gestures & Their GIFs

| Gesture Name    | Trigger Logic                                       | Plays GIF           |
| --------------- | --------------------------------------------------- | ------------------- |
| **tuff**        | Hands at different height + distance between wrists | `tuff.gif`          |
| **thinking**    | Finger (index tip) near mouth                       | `thinking.gif`      |
| **idea**        | Index finger pointing upwards                       | `idea.gif`          |
| **blink / nod** | Very fast downward head movement                    | `blink.gif`         |
| **tongue**      | Mouth open beyond threshold                         | `tongue.gif`        |
| **hands_up**    | Both hands raised above nose                        | `hands_up.gif`      |
| **rubbing**     | Both index fingers close to each other              | `rubbing_hands.gif` |
| **neutral**     | Nothing detected                                    | Black screen        |

---

## 🧠 How Detection Works

The script uses **MediaPipe Holistic** to capture:

* 478 **face landmarks**
* 21 **hand landmarks** (per hand)
* Nose movement history for **quick nod detection**
* Eye aspect ratio + mouth gap detection

Gesture stabilization ensures the system does not switch GIFs rapidly.

---

## 🖼 GIF Engine Logic

* GIFs are pre-loaded once at start.
* Each gesture plays **from frame 0 to end**, looped.
* If gesture changes → GIF resets.
* Uses `cv2.addWeighted()` for smooth blending transitions.

---

## ⚠️ Troubleshooting

### ❗ GIF not loading

Check that all GIFs are inside `/assets` folder, correct names:

```
tuff.gif
thinking.gif
idea.gif
blink.gif
tongue.gif
hands_up.gif
rubbing_hands.gif
```

### ❗ Camera not opening

Try changing:

```python
cap = cv2.VideoCapture(0)
```

to:

```python
cap = cv2.VideoCapture(1)
```

### ❗ Slow GIF playback

Use smaller dimensions or reduce FPS from:

```python
gif_fps = 12
```

---

## 📘 Notes

* This project requires good lighting for best detection.
* Works best when camera is at face level.
* Some gestures may conflict — stabilization reduces false positives.

---


Developed by **Chan** with help from ChatGPT.
