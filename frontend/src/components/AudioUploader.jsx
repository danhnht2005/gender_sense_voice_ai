import { useCallback, useState } from "react";

import "../styles/AudioUploader.scss";

const isAudioFile = (file) => {
  const validTypes = [
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/x-wav",
    "audio/wave",
  ];
  return (
    validTypes.includes(file.type) ||
    file.name.match(/\.(wav|mp3|ogg|flac|webm)$/i)
  );
};

export default function AudioUploader({ onFileSelect, disabled }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && isAudioFile(file)) {
        onFileSelect(file);
      }
    },
    [onFileSelect],
  );

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleClick = () => {
    document.getElementById("audio-file-input").click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && isAudioFile(file)) {
      onFileSelect(file);
    }
    e.target.value = "";
  };

  return (
    <div
      className={`uploader-zone glass-card ${isDragging ? "uploader-dragging" : ""} ${disabled ? "uploader-disabled" : ""}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={disabled ? undefined : handleClick}
      id="audio-uploader"
    >
      <input
        type="file"
        id="audio-file-input"
        accept="audio/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
        disabled={disabled}
      />

      <div className="uploader-icon">
        <svg
          width="48"
          height="48"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>

      <div className="uploader-text">
        <p className="uploader-title">
          {isDragging ? "Thả file vào đây..." : "Kéo & thả file audio vào đây"}
        </p>
        <p className="uploader-hint">
          hoặc click để chọn file (.wav, .mp3, .flac)
        </p>
      </div>
    </div>
  );
}
