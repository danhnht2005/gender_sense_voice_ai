# 🎤 Gender Voice AI — Phân Loại Giới Tính Qua Giọng Nói

> Ứng dụng web phân loại giới tính (Nam/Nữ) qua giọng nói sử dụng Deep Learning với kiến trúc **TCN + Transformer + Attention**.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)
![React](https://img.shields.io/badge/React-19-blue?logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Kiến Trúc Mô Hình](#-kiến-trúc-mô-hình)
- [Kết Quả](#-kết-quả)
- [Cấu Trúc Project](#-cấu-trúc-project)
- [Cài Đặt](#-cài-đặt)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [API Endpoints](#-api-endpoints)
- [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)
- [Hạn Chế & Hướng Phát Triển](#-hạn-chế--hướng-phát-triển)

---

## 🎯 Giới Thiệu

**Gender Voice AI** là đồ án môn Deep Learning, xây dựng hệ thống end-to-end cho bài toán **phân loại giới tính qua giọng nói**:

1. **Người dùng** upload file audio hoặc ghi âm trực tiếp từ microphone
2. **Backend** tiền xử lý audio → trích xuất đặc trưng MFCC → đưa qua mô hình Deep Learning
3. **Kết quả** trả về: giới tính dự đoán (Nam/Nữ), xác suất, thời gian xử lý

### Đặc điểm nổi bật

- 🧠 Kiến trúc hybrid **TCN + Transformer + Attention** (~118K tham số)
- 🎯 Độ chính xác **100%** trên tập test (2,130 mẫu)
- ⚡ Thời gian inference **< 50ms** trên CPU
- 🎙️ Hỗ trợ upload file (.wav, .mp3, .flac, .webm) và ghi âm trực tiếp
- 🌐 Giao diện web hiện đại với dark theme và glassmorphism

---

## 🧠 Kiến Trúc Mô Hình

```
Input Audio (3s, 16kHz)
    │
    ▼
┌─────────────────────────────────┐
│  Tiền xử lý                    │
│  DC Offset → Normalize →       │
│  Preemphasis → Resample 16kHz  │
│  → Pad/Trim 3s                 │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trích xuất đặc trưng          │
│  MFCC(20) + Delta + Delta²     │
│  = 60 features × 94 frames     │
└─────────────────────────────────┘
    │
    ▼  (batch, 94, 60)
┌─────────────────────────────────┐
│  TCN Block ×3                   │
│  Dilated Conv (d=1,2,4)         │
│  + BatchNorm + ReLU + Residual  │
└─────────────────────────────────┘
    │
    ▼  (batch, 94, 64)
┌─────────────────────────────────┐
│  Positional Encoding            │
│  Sinusoidal (từ Attention Is    │
│  All You Need)                  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Transformer Block ×1           │
│  Multi-Head Self-Attention      │
│  (4 heads) + Feed-Forward       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Attention Aggregation          │
│  Learnable Soft Attention       │
│  → Weighted sum over time       │
└─────────────────────────────────┘
    │
    ▼  (batch, 64)
┌─────────────────────────────────┐
│  Classifier                     │
│  FC(64→32) → ReLU → FC(32→1)   │
│  → Sigmoid → Male/Female       │
└─────────────────────────────────┘
```

**Tổng tham số:** 117,826 (trainable)

---

## 📊 Kết Quả

### Mô hình chính (TCN + Transformer + Attention)

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 100.0% |
| **Precision** | 100.0% |
| **Recall** | 100.0% |
| **F1-Score** | 100.0% |
| **AUC** | 1.0 |

### So sánh với Baseline

| Mô hình | Accuracy | F1-Score | AUC | Tham số |
|---------|----------|----------|-----|---------|
| Random Forest | 96.62% | 95.79% | 0.994 | N/A |
| SVM (RBF) | 99.44% | 99.31% | 0.999 | N/A |
| Simple 1D-CNN | 100.0% | 100.0% | 1.0 | 26,785 |
| **TCN+Trans+Attn** | **100.0%** | **100.0%** | **1.0** | **117,826** |

### Dataset

- **Nguồn:** https://drive.google.com/drive/folders/16MsyJGC9iFV6CnHuI6q8VCM8lK_WMBIM?usp=drive_link
- **Tổng mẫu:** 14,196 file audio
- **Phân chia:** 70% Train / 15% Validation / 15% Test
- **Tham số audio:** 16kHz, mono, 3 giây/mẫu

---

## 📁 Cấu Trúc Project

```
gender_sense_voice_ai/
│
├── model/                          # 🧠 Deep Learning Pipeline
│   ├── demo.py                     # Tiền xử lý audio + trích xuất MFCC
│   ├── model_tcn_transformer_attention.py  # Kiến trúc mô hình
│   ├── train.py                    # Training + Early Stopping + LR Scheduler
│   ├── evaluate.py                 # Đánh giá mô hình (metrics, plots)
│   ├── baseline.py                 # So sánh với SVM, RF, CNN
│   └── inference.py                # Dự đoán từ command line
│
├── api/                            # 🔵 Backend (FastAPI)
│   ├── main.py                     # Endpoints: /predict, /health, /model-info
│   ├── schemas.py                  # Pydantic models (request/response)
│   ├── audio_processor.py          # Xử lý upload → MFCC (hỗ trợ webm convert)
│   ├── model_loader.py             # Singleton model service
│   └── requirements.txt            # Dependencies
│
├── frontend/                       # 🟢 Frontend (React + Vite)
│   └── src/
│       ├── App.jsx                 # Trang chính
│       ├── components/
│       │   ├── Header.jsx          # Header + logo
│       │   ├── AudioUploader.jsx   # Drag & drop upload
│       │   ├── AudioRecorder.jsx   # Ghi âm từ microphone
│       │   ├── WaveformVisualizer.jsx  # Hiển thị waveform (wavesurfer.js)
│       │   ├── ResultDisplay.jsx   # Kết quả dự đoán
│       │   ├── ModelInfo.jsx       # Thông tin mô hình
│       │   └── Footer.jsx          # Footer
│       └── styles/                 # SCSS styles
│
├── notebooks/                      # 📓 Jupyter Notebooks
│   └── EDA.ipynb                   # Phân tích dữ liệu
│
├── outputs/                        # 📦 Kết quả training
│   ├── best_model.pth              # Trọng số mô hình tốt nhất
│   ├── norm_stats.json             # Mean/Std cho normalization
│   ├── evaluation_summary.json     # Metrics đánh giá
│   ├── training_history.json       # Lịch sử training
│   ├── confusion_matrix.png        # Ma trận nhầm lẫn
│   ├── roc_curve.png               # Đường cong ROC
│   └── training_curves.png         # Loss/Accuracy curves
│
├── dataset/                        # 🎵 Dataset (không đẩy lên GitHub)
│   ├── male/                       # ~8,400 file .wav giọng nam
│   └── female/                     # ~5,800 file .wav giọng nữ
│
├── .gitignore
└── README.md
```

---

## ⚙️ Cài Đặt

### Yêu cầu hệ thống

- Python >= 3.10
- Node.js >= 18
- FFmpeg (tự động cài qua `imageio-ffmpeg`)

### 1. Clone repository

```bash
git clone https://github.com/danhnht2005/gender_sense_voice_ai.git
cd gender_sense_voice_ai
```

### 2. Cài đặt Backend

```bash
# Cài dependencies Python
pip install torch torchvision torchaudio
pip install fastapi uvicorn python-multipart librosa numpy pydub imageio-ffmpeg scikit-learn

# Hoặc dùng requirements.txt
pip install -r api/requirements.txt
pip install pydub imageio-ffmpeg
```

### 3. Cài đặt Frontend

```bash
cd frontend
npm install
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Chạy Backend (FastAPI)

```bash
cd api
python -m uvicorn main:app --reload --port 8000
```

Server sẽ chạy tại: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

### Chạy Frontend (React + Vite)

```bash
cd frontend
npm run dev
```

Giao diện sẽ mở tại: `http://localhost:5173`

### Sử dụng

1. Mở trình duyệt tại `http://localhost:5173`
2. **Upload file audio:** Kéo thả hoặc click chọn file (.wav, .mp3, .flac, .webm)
3. **Ghi âm:** Nhấn "Ghi âm từ Mic" → nói vài giây → nhấn "Dừng"
4. Xem kết quả: giới tính dự đoán, xác suất, waveform, thời gian xử lý

---

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Kiểm tra trạng thái server |
| `GET` | `/model-info` | Thông tin mô hình (params, accuracy, device) |
| `POST` | `/predict` | Upload file audio → dự đoán giới tính |

### Ví dụ gọi API

```bash
# Health check
curl http://localhost:8000/health

# Dự đoán
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/audio.wav"
```

### Response mẫu

```json
{
  "label": "Male",
  "confidence": 0.9998,
  "probability_male": 0.9998,
  "probability_female": 0.0002,
  "processing_time_ms": 45.3
}
```

---

## 🛠 Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|------------|-----------|
| **Deep Learning** | PyTorch 2.11 |
| **Xử lý Audio** | Librosa 0.11, FFmpeg |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | React 19, Vite 8, Axios |
| **Visualization** | WaveSurfer.js |
| **Styling** | SCSS, Glassmorphism, CSS Animations |
| **ML Baselines** | scikit-learn (SVM, Random Forest) |

---

## Training Pipeline

Nếu muốn tự train lại mô hình:

```bash
# Bước 1: Chuẩn bị dataset (đặt file .wav vào dataset/male/ và dataset/female/)

# Bước 2: Tiền xử lý + tạo cache
python model/demo.py

# Bước 3: Training
python model/train.py

# Bước 4: Đánh giá
python model/evaluate.py

# Bước 5: So sánh baseline
python model/baseline.py
```

**Training config:**
- Epochs: 50 (Early Stopping patience=7)
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau
- Batch size: 32
- Split: 70/15/15 (stratified)
- Class weights: tự động cân bằng

---

## ⚠️ Hạn Chế & Hướng Phát Triển

### Hạn chế hiện tại

- Dataset từ CMU Arctic (thu studio, chất lượng cao) → chưa đại diện cho giọng nói thực tế
- Chưa xử lý tốt các trường hợp edge case: giọng trẻ em, giọng chuyển giới, giọng nói có nhiễu nền mạnh
- Model chạy trên CPU (chưa tối ưu cho GPU inference)

### Hướng phát triển

- 🔊 Mở rộng dataset với VoxCeleb, CommonVoice để tăng tính đa dạng
- 🌍 Thêm phân loại đa ngôn ngữ (tiếng Việt, tiếng Anh, tiếng Trung...)
- 📱 Tối ưu model cho mobile (quantization, ONNX export)
- 🐳 Containerize bằng Docker để deploy dễ dàng hơn
- 🔄 Thêm real-time streaming inference

---

## 👥 Thông Tin

- **Đồ án:** Deep Learning — Học kỳ 6 (2025-2026)
- **Công nghệ:** PyTorch • FastAPI • React • TCN + Transformer + Attention
