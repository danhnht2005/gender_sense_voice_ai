# Tổng hợp Kiến trúc & Tham số Mô hình TCN + Transformer + Attention

Dưới đây là tổng hợp chi tiết về tham số và kiến trúc của mô hình đang được sử dụng trong dự án (định nghĩa tại file `model_tcn_transformer_attention.py`):

## 1. Tổng quan mô hình
* **Tên mô hình:** `TCN_Transformer_Attention_Model`
* **Mục đích:** Phân loại giới tính qua giọng nói (Binary classification: Nam/Nữ)
* **Tổng số tham số (Trainable Parameters):** Khoảng **118.000** (~118K) tham số.

## 2. Các siêu tham số (Hyperparameters) mặc định
* **`input_dim = 60`**: Số lượng đặc trưng đầu vào (tương ứng với 60 MFCC features).
* **`embed_dim = 64`**: Số chiều không gian nhúng (embedding dimension).
* **`tcn_channels = 64`**: Số lượng kênh (filters) trong các khối TCN.
* **`num_tcn = 3`**: Số lượng khối TCN (Temporal Convolutional Network).
* **`num_transformer = 1`**: Số lượng khối Transformer Encoder.
* **`num_heads = 4`**: Số lượng "đầu" tập trung (attention heads) trong Multi-Head Attention.
* **`dropout = 0.2`**: Tỉ lệ loại bỏ ngẫu nhiên nơ-ron để tránh hiện tượng học vẹt (overfitting).

## 3. Cấu trúc và luồng dữ liệu (Pipeline)
Đầu vào có định dạng tensor **(batch, 94, 60)** tương ứng với *T=94* khung thời gian và *F=60* đặc trưng. Dữ liệu đi qua các lớp sau:

1. **Khối TCN (Causal Temporal Convolutional Network):** 
   * Gồm 3 khối liên tiếp có hệ số giãn nở (`dilation`) tăng dần theo lũy thừa: **1, 2 và 4**.
   * Sử dụng Causal Convolution kết hợp `Chomp1d` để đảm bảo mô hình chỉ học từ quá khứ, không nhìn thấy tương lai.
   * Nhiệm vụ: Trích xuất các đặc trưng cục bộ (local features).

2. **Mã hóa vị trí (Positional Encoding):**
   * Sử dụng thuật toán Sinusoidal để bổ sung thông tin vị trí thời gian của từng khung âm thanh, giúp Transformer hiểu được thứ tự các âm tiết.

3. **Khối Transformer:**
   * Sử dụng cơ chế Self-Attention (`4 heads`) để học các mối quan hệ ngữ cảnh toàn cục xuyên suốt độ dài của đoạn âm thanh (hiểu được mối quan hệ ở cự ly xa).
   * Lớp Feed-Forward bên trong sử dụng kích thước chiều ẩn `ff_dim = 128` (gấp đôi `embed_dim`).

4. **Cơ chế Soft Attention (Attention Aggregation):**
   * Thay vì sử dụng bộ gộp trung bình thông thường (average pooling), lớp Attention này học cách gán trọng số cho từng khung thời gian, dồn toàn bộ sự tập trung vào những phần âm thanh chứa đặc trưng giới tính rõ rệt nhất.

5. **Đầu ra phân loại (Classification Head):**
   * Một mạng kết nối đầy đủ (Fully Connected) thu nhỏ kích thước: `Linear(64, 32)` -> `BatchNorm` -> `ReLU` -> `Dropout` -> `Linear(32, 1)`.
   * Đầu ra cuối cùng là một giá trị **logit duy nhất** thể hiện xác suất của giới tính.
