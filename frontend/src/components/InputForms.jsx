import { useState } from "react";

export default function InputForms({ onSubmit, loading }) {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim() || !text.trim()) return;
    onSubmit({ type: "text", title: title.trim(), text: text.trim() });
  };

  const isDisabled = loading || !title.trim() || !text.trim();

  return (
    <form className="input-forms" onSubmit={handleSubmit}>
      {/* Form body */}
      <div className="tab-body">
        <div className="field">
          <label htmlFor="news-title">Headline / Title</label>
          <input
            id="news-title"
            type="text"
            placeholder="e.g. Breaking: Government announces new policy"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            autoComplete="off"
          />
        </div>
        <div className="field">
          <label htmlFor="news-text">Article Body</label>
          <textarea
            id="news-text"
            rows={7}
            placeholder="Paste the full article text here…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        className={`submit-btn ${loading ? "loading" : ""}`}
        disabled={isDisabled}
        id="analyze-btn"
      >
        {loading ? (
          <span className="spinner" />
        ) : (
          <>
            <span className="btn-icon">🔍</span> Analyze
          </>
        )}
      </button>
    </form>
  );
}
