import React, { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function App() {
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    if (!file.name.endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }

    setError(null);
    setUploading(true);
    setUploadProgress(0);
    setMessages([]);
    setSession(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          setUploadProgress(Math.round((e.loaded * 100) / e.total));
        },
      });

      setSession(res.data);
      setMessages([
        {
          role: "assistant",
          content: `I've analyzed **${res.data.filename}** and indexed it into **${res.data.num_chunks} semantic chunks**.\n\nYou can now ask me anything about this document — summaries, specific details, comparisons, or explanations.`,
          sources: [],
          time: null,
        },
      ]);
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed. Please try again.");
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    disabled: uploading || thinking,
  });

  const sendMessage = async () => {
    const q = input.trim();
    if (!q || !session || thinking) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setThinking(true);
    setError(null);

    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        session_id: session.session_id,
        question: q,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.data.answer,
          sources: res.data.sources,
          time: res.data.processing_time,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "⚠️ Something went wrong. Please try again.",
          sources: [],
          time: null,
          isError: true,
        },
      ]);
    } finally {
      setThinking(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resetSession = () => {
    if (session) {
      axios.delete(`${API_BASE}/session/${session.session_id}`).catch(() => {});
    }
    setSession(null);
    setMessages([]);
    setInput("");
    setError(null);
  };

  const suggestedQuestions = [
    "Summarize the key points of this document",
    "What are the main conclusions?",
    "List the most important findings",
    "What topics does this document cover?",
  ];

  return (
    <div className="app">
      {/* Ambient background */}
      <div className="ambient" />

      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">DocMind</span>
          </div>
          <span className="logo-tagline">PDF Intelligence Engine</span>
        </div>
        {session && (
          <div className="header-right">
            <div className="doc-badge">
              <span className="doc-badge-icon">📄</span>
              <span className="doc-badge-name">{session.filename}</span>
              <span className="doc-badge-chunks">{session.num_chunks} chunks</span>
            </div>
            <button className="btn-ghost" onClick={resetSession}>
              New Document
            </button>
          </div>
        )}
      </header>

      {/* Main */}
      <main className="main">
        {!session ? (
          /* Upload Screen */
          <div className="upload-screen">
            <div className="upload-hero">
              <h1 className="hero-title">
                Ask anything about<br />
                <span className="hero-accent">your documents</span>
              </h1>
              <p className="hero-sub">
                Upload a PDF and get instant, accurate answers powered by RAG + GPT-4.
              </p>
            </div>

            <div
              {...getRootProps()}
              className={`dropzone ${isDragActive ? "dropzone--active" : ""} ${uploading ? "dropzone--uploading" : ""}`}
            >
              <input {...getInputProps()} />
              {uploading ? (
                <div className="upload-progress">
                  <div className="spinner" />
                  <p className="upload-status">Processing PDF...</p>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <span className="progress-pct">{uploadProgress}%</span>
                </div>
              ) : (
                <div className="dropzone-content">
                  <div className="dropzone-icon">⬆</div>
                  <p className="dropzone-title">
                    {isDragActive ? "Drop your PDF here" : "Drop your PDF here"}
                  </p>
                  <p className="dropzone-sub">or click to browse · Max 20MB</p>
                </div>
              )}
            </div>

            {error && <div className="error-banner">⚠ {error}</div>}

            <div className="feature-grid">
              {[
                { icon: "⚡", title: "Low-Latency Retrieval", desc: "FAISS vector search finds relevant chunks in milliseconds" },
                { icon: "🧠", title: "Context-Aware Answers", desc: "Conversational memory across follow-up questions" },
                { icon: "📍", title: "Source Attribution", desc: "Every answer cites exact page numbers from your document" },
                { icon: "🔍", title: "MMR Search", desc: "Maximal Marginal Relevance reduces redundant retrieved chunks" },
              ].map((f) => (
                <div key={f.title} className="feature-card">
                  <span className="feature-icon">{f.icon}</span>
                  <h3 className="feature-title">{f.title}</h3>
                  <p className="feature-desc">{f.desc}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          /* Chat Screen */
          <div className="chat-screen">
            <div className="messages">
              {messages.map((msg, i) => (
                <div key={i} className={`message message--${msg.role} ${msg.isError ? "message--error" : ""}`}>
                  <div className="message-avatar">
                    {msg.role === "assistant" ? "◈" : "U"}
                  </div>
                  <div className="message-body">
                    <div className="message-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources">
                        <span className="sources-label">Sources</span>
                        {msg.sources.map((s, j) => (
                          <div key={j} className="source-chip">
                            <span className="source-page">p.{s.page}</span>
                            <span className="source-preview">{s.preview}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {msg.time && (
                      <span className="message-meta">{msg.time}s</span>
                    )}
                  </div>
                </div>
              ))}

              {thinking && (
                <div className="message message--assistant">
                  <div className="message-avatar">◈</div>
                  <div className="message-body">
                    <div className="thinking">
                      <span /><span /><span />
                    </div>
                  </div>
                </div>
              )}

              {/* Suggested questions shown only initially */}
              {messages.length === 1 && (
                <div className="suggestions">
                  {suggestedQuestions.map((q) => (
                    <button
                      key={q}
                      className="suggestion-btn"
                      onClick={() => { setInput(q); inputRef.current?.focus(); }}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input bar */}
            <div className="input-area">
              {error && <div className="error-inline">⚠ {error}</div>}
              <div className="input-bar">
                <textarea
                  ref={inputRef}
                  className="input-field"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question about your document..."
                  rows={1}
                  disabled={thinking}
                />
                <button
                  className={`send-btn ${(!input.trim() || thinking) ? "send-btn--disabled" : ""}`}
                  onClick={sendMessage}
                  disabled={!input.trim() || thinking}
                >
                  ↑
                </button>
              </div>
              <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
