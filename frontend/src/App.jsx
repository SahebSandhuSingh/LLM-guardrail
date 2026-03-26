import { useState, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:8000/analyze";
const SESSION_ID = "demo-" + Date.now();

function riskClass(level) {
  if (level === "critical") return "risk-critical";
  if (level === "high") return "risk-high";
  if (level === "medium") return "risk-medium";
  return "risk-low";
}

function barColor(score) {
  if (score >= 0.7) return "#ef4444";
  if (score >= 0.4) return "#f59e0b";
  return "#22c55e";
}

const CATEGORY_NAMES = {
  prompt_injection: "Prompt Injection",
  jailbreak: "Jailbreak",
  pii_extraction: "PII Extraction",
  persona_hijack: "Persona Hijack",
};

export default function App() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  async function handleSend() {
    if (!message.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, message: message.trim() }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResponse(data);
      setHistory((prev) => [
        ...prev,
        { message: message.trim(), risk: data.session_risk },
      ]);
      setMessage("");
    } catch {
      setError("Backend not reachable");
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && message.trim()) handleSend();
  }

  const explanation = response?.explanation || {};
  const action = response?.action || {};
  const scores = response?.category_scores || {};
  const riskLevel = explanation.risk_level || "low";

  return (
    <div className="app">
      <h1>🛡️ Sentinel Guard Demo</h1>

      {/* ---------- Input Panel ---------- */}
      <div className="input-panel">
        <input
          ref={inputRef}
          type="text"
          placeholder="Type a message to analyze…"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button onClick={handleSend} disabled={!message.trim() || loading}>
          {loading ? "…" : "Send"}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {loading && <div className="loading">Analyzing…</div>}

      {!response && !loading && (
        <div className="empty-state">Send a message to see the analysis.</div>
      )}

      {response && !loading && (
        <>
          {/* ---------- Session Risk ---------- */}
          <div className="card">
            <h2>Session Risk</h2>
            <div className="risk-display">
              <div className={`risk-score ${riskClass(riskLevel)}`}>
                {response.session_risk.toFixed(4)}
              </div>
              <div className={`risk-level ${riskClass(riskLevel)}`}>
                {riskLevel}
              </div>
            </div>
          </div>

          {/* ---------- Action Taken ---------- */}
          <div className="card">
            <h2>Action Taken</h2>
            <div className="action-grid">
              <div>
                <div className="label">Stage</div>
                <div className="value">{action.stage}</div>
              </div>
              <div>
                <div className="label">Decision</div>
                <div className="value">{action.decision}</div>
              </div>
            </div>
            {action.response_message && (
              <div className="action-message">{action.response_message}</div>
            )}
          </div>

          {/* ---------- Category Scores ---------- */}
          <div className="card">
            <h2>Category Scores</h2>
            <div className="score-bars">
              {Object.entries(CATEGORY_NAMES).map(([key, label]) => {
                const val = scores[key] ?? 0;
                return (
                  <div className="score-row" key={key}>
                    <span className="name">{label}</span>
                    <div className="score-bar-bg">
                      <div
                        className="score-bar-fill"
                        style={{
                          width: `${Math.max(val * 100, 1)}%`,
                          background: barColor(val),
                        }}
                      />
                    </div>
                    <span className="num">{val.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ---------- Explanation ---------- */}
          <div className="card explanation">
            <h2>Explanation</h2>
            <p className="summary">{explanation.summary}</p>
            <p>
              <strong>Primary threat:</strong>{" "}
              {CATEGORY_NAMES[explanation.primary_threat] ||
                explanation.primary_threat ||
                "none"}
            </p>
            {explanation.contributing_factors?.length > 0 && (
              <>
                <p style={{ marginTop: 8 }}>
                  <strong>Contributing factors:</strong>
                </p>
                <ul className="factors">
                  {explanation.contributing_factors.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </>
            )}
          </div>

          {/* ---------- Session History ---------- */}
          {history.length > 0 && (
            <div className="card">
              <h2>Session History</h2>
              <div className="history-list">
                {history.map((h, i) => (
                  <div className="history-item" key={i}>
                    <span className="msg">
                      Message {i + 1}: "{h.message}"
                    </span>
                    <span className={riskClass(
                      h.risk < 0.3 ? "low" : h.risk < 0.6 ? "medium" : h.risk < 0.85 ? "high" : "critical"
                    )}>
                      {h.risk.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
