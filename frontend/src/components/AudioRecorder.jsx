import { useState, useRef, useCallback } from "react";

import "../styles/AudioRecorder.scss";

export default function AudioRecorder({ onRecordingComplete, disabled }) {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        onRecordingComplete(file);
        stream.getTracks().forEach((track) => track.stop());
        clearInterval(timerRef.current);
        setDuration(0);
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Timer
      let sec = 0;
      timerRef.current = setInterval(() => {
        sec++;
        setDuration(sec);
      }, 1000);
    } catch (err) {
      alert("Không thể truy cập microphone. Vui lòng cấp quyền truy cập.");
      console.error("Microphone error:", err);
    }
  }, [onRecordingComplete]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60)
      .toString()
      .padStart(2, "0");
    const s = (sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  return (
    <div
      className={`recorder glass-card ${isRecording ? "recorder-active" : ""} ${disabled ? "uploader-disabled" : ""}`}
    >
      {!isRecording ? (
        <button
          className="btn btn-primary recorder-btn"
          onClick={startRecording}
          disabled={disabled}
          id="record-button"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" y1="19" x2="12" y2="23" />
            <line x1="8" y1="23" x2="16" y2="23" />
          </svg>
          Ghi âm từ Mic
        </button>
      ) : (
        <div className="recorder-recording">
          <div className="recorder-pulse"></div>
          <span className="recorder-timer">{formatTime(duration)}</span>
          <span className="recorder-label">Đang ghi âm...</span>
          <button
            className="btn btn-danger"
            onClick={stopRecording}
            id="stop-record-button"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <rect x="4" y="4" width="16" height="16" rx="2" />
            </svg>
            Dừng
          </button>
        </div>
      )}
    </div>
  );
}
