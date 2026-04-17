import { useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Label,
} from "recharts";

const COLORS = {
  Real: ["#22c55e", "#166534"],   // green
  Fake: ["#ef4444", "#991b1b"],   // red
};

function ConfidenceGauge({ prediction, confidence }) {
  const pct = Math.round(confidence * 100);
  const colors = COLORS[prediction] || COLORS.Fake;

  const data = [
    { name: prediction, value: pct },
    { name: "rest", value: 100 - pct },
  ];

  return (
    <div className="gauge-wrapper">
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            innerRadius={70}
            outerRadius={95}
            startAngle={180}
            endAngle={0}
            paddingAngle={2}
            dataKey="value"
            stroke="none"
          >
            <Cell fill={colors[0]} />
            <Cell fill="rgba(255,255,255,0.08)" />
            <Label
              value={`${pct}%`}
              position="center"
              dy={-6}
              style={{
                fontSize: "2rem",
                fontWeight: 700,
                fill: colors[0],
              }}
            />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <p className="gauge-label">Confidence</p>
    </div>
  );
}



export default function ResultDisplay({ result, error }) {
  if (error) {
    return (
      <div className="result-card error-card fade-in">
        <div className="error-icon">⚠️</div>
        <h3>Analysis Failed</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!result) return null;

  const isReal = result.prediction === "Real";

  return (
    <div className={`result-card fade-in ${isReal ? "real" : "fake"}`}>
      {/* Badge */}
      <div className={`verdict-badge ${isReal ? "badge-real" : "badge-fake"}`}>
        <span className="verdict-emoji">{isReal ? "✅" : "🚨"}</span>
        <span className="verdict-text">
          {isReal ? "Likely Real" : "Likely Fake"}
        </span>
      </div>

      {/* Gauge */}
      <ConfidenceGauge
        prediction={result.prediction}
        confidence={result.confidence}
      />


    </div>
  );
}
