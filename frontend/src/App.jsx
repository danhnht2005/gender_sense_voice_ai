import { useState, useCallback } from "react";
import axios from "axios";

import Header from "./components/Header";
import Footer from "./components/Footer";
import AudioUploader from "./components/AudioUploader";

import "./App.css";

const API_URL = "http://localhost:8000";

export default function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = useCallback(async (file) => {
    setAudioFile(file);
    setResult(null);
    setError(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(
        err.response?.data?.detail ||
          "Khong the ket noi toi server. Vui long kiem tra API dang chay.",
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <>
      <Header />

      <main className="main">
        <div className="container">
          <div className="input-section section">
            <div className="input-grid">
              <AudioUploader
                onFileSelect={handlePredict}
                disabled={isLoading}
              />
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </>
  );
}
