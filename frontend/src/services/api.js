/**
 * API service — Axios wrapper for the Fake News Detection backend.
 */
import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  // Grey-area predictions may trigger a web-search + LLM fact-check
  // which can take longer than a pure ML inference.
  timeout: 120000,
  headers: { "Content-Type": "application/json" },
});

/**
 * Predict from raw title + text.
 */
export async function predictText(title, text) {
  const { data } = await api.post("/predict/text", { title, text });
  return data;
}

/**
 * Health-check.
 */
export async function healthCheck() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
