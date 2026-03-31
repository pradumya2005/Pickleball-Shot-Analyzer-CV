# Pickleball-Shot-Analyzer-CV
PickleballVision: An AI-powered biomechanical analysis tool for Pickleball coaches. Uses MediaPipe Pose Estimation to automatically detect and classify Dinks and Smashes from video footage using joint angle trigonometry and spatial tracking.
# 🎾 PickleballVision: Biomechanical Shot Analyzer
**Course:** Computer Vision (BYOP Project)  
**Developer:** Pradumya Salunke  

## 🌟 Overview
PickleballVision is a Computer Vision application designed to provide objective technical feedback to Pickleball players. By analyzing video footage, the system tracks human skeletal landmarks to distinguish between high-extension **Smashes** and low-control **Dinks**.

This project solves a real-world problem observed in the Pickleball : the lack of accessible, data-driven coaching tools for amateur and intermediate players.



## 🛠️ Technical Features
- **Pose Estimation:** Utilizes the MediaPipe Tasks API for 3D landmark tracking.
- **Biomechanical Logic:** Calculates the interior angle of the elbow joint using the Law of Cosines and Atan2.
- **Shot Classification:** - **Dink:** Detected when the wrist is below a 0.75 normalized Y-threshold.
  - **Smash:** Detected when the wrist is above 0.35 Y-threshold and the elbow angle is $> 155^\circ$.
- **Video Processing:** Batch processes `.mp4` files and exports an annotated analysis video.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.12+
- OpenCV
- MediaPipe
- NumPy

### 2. Installation
```bash
pip install -r requirements.txt
