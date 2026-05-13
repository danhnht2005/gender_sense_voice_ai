import { useState, useEffect } from "react";
import axios from "axios";

import "../styles/ModelInfo.scss";

const API_URL = "http://localhost:8000";

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    axios
      .get(`${API_URL}/model-info`)
      .then((res) => setInfo(res.data))
      .catch((err) => console.error("Failed to load model info:", err));
  }, []);

  if (!info) return null;

  return (
    <div className="model-info section glass-card">
      <button
        className="model-info-toggle"
        onClick={() => setIsOpen(!isOpen)}
        id="model-info-toggle"
      >
        <div className="model-info-toggle-left">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--accent-purple)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.58-3.25 3.93V22h-1.5V9.93C9.4 9.58 8 7.95 8 6a4 4 0 0 1 4-4z" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="18" r="3" />
          </svg>
          <span>Thông tin mô hình</span>
        </div>
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          style={{
            transform: isOpen ? "rotate(180deg)" : "rotate(0)",
            transition: "transform 0.3s ease",
          }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {isOpen && (
        <div className="model-info-content animate-slide-down">
          <div className="model-info-grid">
            <div className="model-info-item">
              <span className="model-info-label">Kiến trúc</span>
              <span className="model-info-value">
                {info.model_name
                  .replace("_Model", "")
                  .replace(/_/g, " + ")}
              </span>
            </div>
            <div className="model-info-item">
              <span className="model-info-label">Tham số</span>
              <span className="model-info-value">
                {info.total_parameters.toLocaleString()}
              </span>
            </div>
            <div className="model-info-item">
              <span className="model-info-label">Best Epoch</span>
              <span className="model-info-value">{info.best_epoch}</span>
            </div>
            <div className="model-info-item">
              <span className="model-info-label">Độ chính xác</span>
              <span
                className="model-info-value"
                style={{ color: "var(--accent-green)" }}
              >
                {(info.training_accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="model-info-item">
              <span className="model-info-label">Đặc trưng</span>
              <span className="model-info-value">{info.features}</span>
            </div>
            <div className="model-info-item">
              <span className="model-info-label">Thiết bị</span>
              <span className="model-info-value">
                {info.device.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
