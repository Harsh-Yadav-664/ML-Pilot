import React from 'react';

const Dashboard: React.FC = () => (
  <div className="p-8">
    <h1 className="text-3xl font-bold text-gray-900 mb-2">MLPilot Dashboard</h1>
    <p className="text-gray-500 mb-8">Evidence-driven ML experimentation platform.</p>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {[{label: 'Projects', value: '—'}, {label: 'Experiments', value: '—'}, {label: 'Models', value: '—'}].map(card => (
        <div key={card.label} className="bg-white rounded-xl shadow p-6 flex flex-col items-center">
          <span className="text-4xl font-bold text-indigo-600">{card.value}</span>
          <span className="text-gray-500 mt-2">{card.label}</span>
        </div>
      ))}
    </div>
    <div className="mt-10 bg-indigo-50 border border-indigo-200 rounded-xl p-6">
      <h2 className="font-semibold text-indigo-800 mb-1">Phase 0 — Skeleton</h2>
      <p className="text-indigo-700 text-sm">All interfaces, schemas, and providers are ready. Phase 1 will wire training, metrics, and the AI agent loop.</p>
    </div>
  </div>
);

export default Dashboard;
