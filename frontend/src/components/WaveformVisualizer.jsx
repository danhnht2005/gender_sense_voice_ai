import { useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";

import "../styles/WaveformVisualizer.scss";

export default function WaveformVisualizer({ audioFile }) {
  const containerRef = useRef(null);
  const wavesurferRef = useRef(null);

  useEffect(() => {
    if (!audioFile || !containerRef.current) return;

    // Destroy previous instance
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
    }

    // Create WaveSurfer
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(0, 229, 255, 0.4)",
      progressColor: "#a855f7",
      cursorColor: "#00e5ff",
      cursorWidth: 2,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 80,
      responsive: true,
      backend: "WebAudio",
    });

    wavesurferRef.current = ws;

    // Load file
    const url = URL.createObjectURL(audioFile);
    ws.load(url);

    ws.on("ready", () => {
      // Auto cleanup object URL after load
    });

    // Click to play/pause
    ws.on("interaction", () => {
      ws.playPause();
    });

    return () => {
      ws.destroy();
      URL.revokeObjectURL(url);
    };
  }, [audioFile]);

  if (!audioFile) return null;

  return (
    <div className="waveform section glass-card">
      <div className="waveform-header">
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--accent-cyan)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
        </svg>
        <span>{audioFile.name}</span>
        <span className="waveform-hint">Click de phat/dung</span>
      </div>
      <div ref={containerRef} className="waveform-canvas"></div>
    </div>
  );
}
