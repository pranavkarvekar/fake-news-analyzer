import { useState } from "react";
import InputForms from "./components/InputForms";
import ResultDisplay from "./components/ResultDisplay";
import { predictText } from "./services/api";

export default function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (payload) => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      let data;
      if (payload.type === "text") {
        data = await predictText(payload.title, payload.text);
      }
      setResult(data);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Something went wrong. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* Decorative background orbs */}
      <div className="bg-orb orb-1" aria-hidden="true" />
      <div className="bg-orb orb-2" aria-hidden="true" />
      <div className="bg-orb orb-3" aria-hidden="true" />

      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🛡️</span>
            <h1>
              Fake News <span className="accent">Detector</span>
            </h1>
          </div>
          <p className="subtitle">
            ML-powered analysis &middot; 93.5% accuracy &middot; Instant
            results
          </p>
        </div>
      </header>

      <main className="main">
        <section className="card glass" id="input-section">
          <InputForms onSubmit={handleSubmit} loading={loading} />
        </section>

        <section className="card glass" id="result-section">
          {!result && !error && !loading && (
            <div className="empty-state fade-in">
              <span className="empty-icon">📰</span>
              <p>
                Paste a news article above and hit <strong>Analyze</strong>{" "}
                to check if it's real or fake.
              </p>
            </div>
          )}
          {loading && (
            <div className="empty-state fade-in">
              <span className="spinner large" />
              <p>Analyzing article…</p>
            </div>
          )}
          <ResultDisplay result={result} error={error} />
        </section>
      </main>

      <footer className="footer">
        <p>
          Built with FastAPI &amp; React &middot; Model trained on{" "}
          <a
            href="https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification"
            target="_blank"
            rel="noreferrer"
          >
            WELFake Dataset
          </a>
        </p>
      </footer>
    </div>
  );
}
