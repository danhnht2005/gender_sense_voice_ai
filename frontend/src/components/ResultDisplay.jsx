export default function ResultDisplay({ result, isLoading }) {
  if (isLoading) {
    return (
      <div className="result section glass-card">
        <div className="result-loading">
          <div className="spinner"></div>
          <span>Dang phan tich giong noi...</span>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const isMale = result.label === "Male";
  const confidence = (result.confidence * 100).toFixed(1);
  const probMale = (result.probability_male * 100).toFixed(1);
  const probFemale = (result.probability_female * 100).toFixed(1);

  return (
    <div
      className="result section glass-card animate-slide-up"
      id="prediction-result"
    >
      <h3 className="result-heading">Ket qua du doan</h3>

      {/* Main Prediction */}
      <div
        className={`result-main ${isMale ? "result-male" : "result-female"}`}
      >
        <div className="result-icon">
          {isMale ? (
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <circle cx="10" cy="14" r="7" />
              <line x1="21" y1="3" x2="15" y2="9" />
              <polyline points="15 3 21 3 21 9" />
            </svg>
          ) : (
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <circle cx="12" cy="8" r="7" />
              <line x1="12" y1="15" x2="12" y2="23" />
              <line x1="8" y1="19" x2="16" y2="19" />
            </svg>
          )}
        </div>
        <div className="result-label">
          {isMale ? "Nam (Male)" : "Nu (Female)"}
        </div>
        <div className="result-confidence">{confidence}%</div>
      </div>

      {/* Probability Bars */}
      <div className="result-bars">
        <div className="result-bar-row">
          <span className="result-bar-label">Nam</span>
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{
                width: `${probMale}%`,
                background: "var(--gradient-male)",
              }}
            ></div>
          </div>
          <span className="result-bar-value">{probMale}%</span>
        </div>
        <div className="result-bar-row">
          <span className="result-bar-label">Nu</span>
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{
                width: `${probFemale}%`,
                background: "var(--gradient-female)",
              }}
            ></div>
          </div>
          <span className="result-bar-value">{probFemale}%</span>
        </div>
      </div>

      {/* Processing Time */}
      <div className="result-meta">
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
        <span>Thoi gian xu ly: {result.processing_time_ms.toFixed(1)} ms</span>
      </div>
    </div>
  );
}
