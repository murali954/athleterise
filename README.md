# 🏏 AthleteRise – Real-Time Cover Drive Analysis

> Advanced cricket biomechanics analysis using AI-powered pose estimation

## 🎯 Features

- **Real-time pose tracking** with MediaPipe
- **Biomechanical metrics**: Elbow angle, spine lean, head position
- **Shot phase detection**: Stance → Stride → Downswing → Impact → Follow-through → Recovery
- **AI skill grading**: Beginner / Intermediate / Advanced
- **Annotated video output** with live overlays
- **Performance charts** and detailed reports

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your-username/athleterise.git
cd athleterise
pip install -r requirements.txt
```

### Usage
```bash
# Basic analysis
python cover_drive_analysis_realtime.py

# Custom video
python cover_drive_analysis_realtime.py --input your_video.mp4

# Webcam analysis  
python cover_drive_analysis_realtime.py --source webcam
```

---

## 📂 Output

After running, check the `output/` folder:

```
output/
├── annotated_video.mp4      # Video with overlays
├── evaluation.json          # Performance scores
├── smoothness_chart.png     # Angle progression graph
└── metrics/                 # Detailed CSV data
```

### Sample Results
```json
{
  "overall_score": 78.5,
  "skill_level": "Intermediate",
  "phases": {
    "stance": {"score": 82},
    "impact": {"score": 75}
  }
}
```

---

## ⚙️ Requirements

- Python 3.8+
- Webcam or video file
- 4GB+ RAM for real-time processing

**Dependencies:**
```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
```



---

**⭐ Star this repo if helpful!**
