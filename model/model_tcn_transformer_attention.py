"""
model_tcn_transformer_attention.py — Kiến trúc Model Chính
==========================================================
TCN + Transformer + Attention cho bài toán phân loại giới tính qua giọng nói.

Kiến trúc:
  Input (batch, T, 60)
    → TCN Blocks (dilated=1,2,4): Trích xuất đặc trưng cục bộ
    → Positional Encoding: Thêm thông tin vị trí
    → Transformer Block (4 heads): Học quan hệ toàn cục
    → Attention Aggregation: Tổng hợp theo trục thời gian
    → Fully Connected: Phân loại nhị phân

Tổng params: ~118K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. TCN Block — Temporal Convolutional Network
# ==============================================================================

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block với dilated convolution.
    
    Dilated Conv cho phép mở rộng receptive field mà không tăng số params:
      - dilation=1: nhìn 3 time steps
      - dilation=2: nhìn 5 time steps  
      - dilation=4: nhìn 9 time steps
    
    Bao gồm: Conv1d → BatchNorm → ReLU → Dropout → Residual Connection
    
    Args:
        in_channels  (int)  : Số kênh đầu vào
        out_channels (int)  : Số kênh đầu ra
        kernel_size  (int)  : Kích thước kernel (default: 3)
        dilation     (int)  : Hệ số dilation (default: 1)
        dropout      (float): Tỉ lệ dropout (default: 0.2)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        # Padding để giữ nguyên chiều dài sequence sau convolution
        # padding = (kernel_size - 1) * dilation // 2  (causal-style)
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            dilation=dilation, 
            padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection: nếu in_channels != out_channels, dùng 1x1 conv
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, T) — input tensor
        Returns:
            (batch, out_channels, T) — output tensor
        """
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Cắt/pad nếu cần để khớp chiều dài
        if out.shape[2] != residual.shape[2]:
            min_len = min(out.shape[2], residual.shape[2])
            out = out[:, :, :min_len]
            residual = residual[:, :, :min_len]
        
        return self.relu(out + residual)


# ==============================================================================
# 2. Positional Encoding — Mã hoá vị trí (Sinusoidal)
# ==============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (từ "Attention Is All You Need").
    
    Thêm thông tin vị trí vào embedding, giúp Transformer biết được
    thứ tự thời gian của các frame trong sequence.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model  (int)  : Dimension của model
        max_len  (int)  : Chiều dài tối đa của sequence
        dropout  (float): Dropout rate
    """
    
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Tính PE matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (không train)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, T, d_model)
        Returns:
            (batch, T, d_model) + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# 3. Transformer Block — Self-Attention + Feed-Forward
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block.
    
    Cấu trúc:
      1. Multi-Head Self-Attention → Add & LayerNorm
      2. Feed-Forward Network → Add & LayerNorm
    
    Self-Attention cho phép mỗi time frame "nhìn" toàn bộ sequence,
    học được quan hệ dài hạn giữa các phần của audio.
    
    Args:
        embed_dim   (int)  : Dimension của embedding
        num_heads   (int)  : Số attention heads
        ff_dim      (int)  : Dimension của feed-forward layer
        dropout     (float): Dropout rate
    """
    
    def __init__(self, embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input: (batch, seq, embed)
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, T, embed_dim)
        Returns:
            (batch, T, embed_dim)
        """
        # Self-Attention + Residual + LayerNorm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


# ==============================================================================
# 4. Attention Aggregation — Soft Attention
# ==============================================================================

class Attention(nn.Module):
    """
    Learnable Soft Attention Aggregation.
    
    Thay vì average pooling đơn giản, Attention học cách gán trọng số
    cho mỗi time frame, tập trung vào những phần audio quan trọng nhất
    cho việc phân loại giới tính.
    
    Score = tanh(W @ h + b) → softmax → weighted sum
    
    Args:
        embed_dim (int): Dimension của input
    """
    
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, T, embed_dim)
        Returns:
            (batch, embed_dim) — weighted sum over time
        """
        # Tính attention scores: (batch, T, 1)
        scores = self.attention(x)
        
        # Softmax over time dimension
        weights = F.softmax(scores, dim=1)  # (batch, T, 1)
        
        # Weighted sum: (batch, T, embed_dim) * (batch, T, 1) → sum → (batch, embed_dim)
        context = torch.sum(weights * x, dim=1)
        
        return context


# ==============================================================================
# 5. Model Chính — TCN + Transformer + Attention
# ==============================================================================

class TCN_Transformer_Attention_Model(nn.Module):
    """
    Model chính cho phân loại giới tính qua giọng nói.
    
    Pipeline:
      1. Input: (batch, T=94, F=60) — MFCC features
      2. TCN Blocks (3 blocks, dilations=[1,2,4]):
         - Trích xuất đặc trưng cục bộ từ MFCC
         - Mở rộng receptive field qua dilated convolution
      3. Positional Encoding:
         - Thêm thông tin thứ tự thời gian
      4. Transformer Block (4 heads):
         - Self-attention học quan hệ toàn cục
      5. Attention Aggregation:
         - Soft attention tổng hợp thông tin quan trọng
      6. Fully Connected:
         - Phân loại nhị phân (Male/Female)
    
    Args:
        input_dim   (int)  : Số features đầu vào (60)
        embed_dim   (int)  : Dimension embedding (64)
        num_heads   (int)  : Số attention heads (4)
        tcn_channels(int)  : Số kênh TCN (64)
        num_tcn     (int)  : Số TCN blocks (3)
        num_transformer(int): Số Transformer blocks (1)
        dropout     (float): Dropout rate (0.2)
    """
    
    def __init__(self, input_dim=60, embed_dim=64, num_heads=4,
                 tcn_channels=64, num_tcn=3, num_transformer=1, dropout=0.2):
        super(TCN_Transformer_Attention_Model, self).__init__()
        
        # TCN Blocks với dilation tăng dần: 1, 2, 4
        tcn_blocks = []
        dilations = [2 ** i for i in range(num_tcn)]  # [1, 2, 4]
        
        for i, d in enumerate(dilations):
            in_ch = input_dim if i == 0 else tcn_channels
            tcn_blocks.append(
                TCNBlock(in_ch, tcn_channels, kernel_size=3, dilation=d, dropout=dropout)
            )
        
        self.tcn = nn.Sequential(*tcn_blocks)
        
        # Projection layer: tcn_channels → embed_dim (nếu khác nhau)
        self.projection = nn.Linear(tcn_channels, embed_dim) \
            if tcn_channels != embed_dim else nn.Identity()
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer Blocks
        self.transformers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim=embed_dim * 2, dropout=dropout)
            for _ in range(num_transformer)
        ])
        
        # Attention Aggregation
        self.attention = Attention(embed_dim)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Binary classification (logit)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, T, F) — T=94 time frames, F=60 features
        Returns:
            (batch, 1) — logit cho binary classification
        """
        # x: (batch, T, F) → (batch, F, T) cho Conv1d
        x = x.permute(0, 2, 1)
        
        # TCN: (batch, F, T) → (batch, tcn_channels, T)
        x = self.tcn(x)
        
        # (batch, tcn_channels, T) → (batch, T, tcn_channels)
        x = x.permute(0, 2, 1)
        
        # Projection: (batch, T, tcn_channels) → (batch, T, embed_dim)
        x = self.projection(x)
        
        # Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Blocks
        for transformer in self.transformers:
            x = transformer(x)
        
        # Attention Aggregation: (batch, T, embed_dim) → (batch, embed_dim)
        x = self.attention(x)
        
        # Classification: (batch, embed_dim) → (batch, 1)
        x = self.classifier(x)
        
        return x


# ==============================================================================
# 6. Utility: Đếm tham số
# ==============================================================================

def count_parameters(model):
    """
    Đếm tổng số tham số trainable của model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Tổng số tham số trainable
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_shape=(4, 94, 60)):
    """
    In summary của model: kiến trúc, số params, output shape.
    
    Args:
        model (nn.Module)   : PyTorch model
        input_shape (tuple) : Shape của input tensor
    """
    print("=" * 60)
    print("  MODEL SUMMARY")
    print("=" * 60)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input shape:  {input_shape}")
    
    # Forward pass kiểm tra
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        output = model(dummy)
    
    print(f"Output shape: {tuple(output.shape)}")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Chi tiết từng layer
    print(f"\n{'Layer':<40} {'Params':>10}")
    print("-" * 52)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:<38} {param.numel():>10,}")
    
    print("-" * 52)
    print(f"  {'TOTAL':<38} {count_parameters(model):>10,}")
    print("=" * 60)


# ==============================================================================
# Main: Test forward pass
# ==============================================================================

if __name__ == "__main__":
    print("Testing TCN + Transformer + Attention Model...")
    print()
    
    # Khởi tạo model
    model = TCN_Transformer_Attention_Model(
        input_dim=60,
        embed_dim=64,
        num_heads=4,
        tcn_channels=64,
        num_tcn=3,
        num_transformer=1,
        dropout=0.2
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 94, 60)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")   # (4, 94, 60)
    print(f"Output shape: {output.shape}")         # (4, 1)
    print(f"Total params: {count_parameters(model):,}")
    print()
    
    # In model summary
    model_summary(model)
    
    print("\n[OK] Forward pass OK!")
