import React, { useState } from 'react';
import { Upload, X, Loader2 } from 'lucide-react';
import api, { runBaselineExperiment } from '../api';

interface DatasetUploaderProps {
  onSuccess: (datasetPath: string, targetColumn: string) => void;
  onCancel: () => void;
}

export function DatasetUploader({ onSuccess, onCancel }: DatasetUploaderProps) {
  const [activeTab, setActiveTab] = useState<'csv' | 'sql'>('csv');
  const [file, setFile] = useState<File | null>(null);
  const [connectionString, setConnectionString] = useState('');
  const [query, setQuery] = useState('');
  
  const [columns, setColumns] = useState<string[]>([]);
  const [datasetPath, setDatasetPath] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await api.post('/ui/data/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDatasetPath(res.data.dataset_path);
      setColumns(res.data.columns);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSqlConnect = async () => {
    if (!connectionString || !query) return;
    setLoading(true);
    setError('');
    
    try {
      const res = await api.post('/ui/data/connect-sql', {
        connection_string: connectionString,
        query: query
      });
      setDatasetPath(res.data.dataset_path);
      setColumns(res.data.columns);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'SQL Connection failed');
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = async () => {
    if (datasetPath && targetColumn) {
      setLoading(true);
      try {
        await runBaselineExperiment(datasetPath, targetColumn);
      } catch (err) {
        console.error("Failed to run baseline:", err);
      }
      setLoading(false);
      onSuccess(datasetPath, targetColumn);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/80 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900 p-6 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-zinc-100">Upload Dataset</h2>
          <button onClick={onCancel} className="text-zinc-500 hover:text-zinc-300">
            <X className="h-5 w-5" />
          </button>
        </div>

        {!columns.length ? (
          <div className="space-y-4">
            <div className="flex rounded-lg bg-zinc-800/50 p-1">
              <button
                onClick={() => setActiveTab('csv')}
                className={`flex-1 rounded-md py-1.5 text-sm font-medium transition-colors ${
                  activeTab === 'csv' ? 'bg-zinc-700 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                CSV Upload
              </button>
              <button
                onClick={() => setActiveTab('sql')}
                className={`flex-1 rounded-md py-1.5 text-sm font-medium transition-colors ${
                  activeTab === 'sql' ? 'bg-zinc-700 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                SQL Connect
              </button>
            </div>

            {activeTab === 'csv' ? (
              <>
                <div className="rounded-lg border-2 border-dashed border-zinc-800 p-8 text-center hover:bg-zinc-800/30 transition-colors">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer text-sm text-zinc-400">
                    <Upload className="mx-auto mb-2 h-8 w-8 text-zinc-500" />
                    {file ? file.name : 'Select a CSV file'}
                  </label>
                </div>
                <button
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="flex w-full items-center justify-center rounded-lg bg-emerald-500 py-2 text-sm font-medium text-zinc-950 transition-colors hover:bg-emerald-400 disabled:opacity-50"
                >
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Upload & Analyze'}
                </button>
              </>
            ) : (
              <div className="space-y-3">
                <div>
                  <label className="mb-1 block text-xs font-medium text-zinc-400">Connection String</label>
                  <input
                    type="text"
                    value={connectionString}
                    onChange={(e) => setConnectionString(e.target.value)}
                    placeholder="postgresql://user:pass@localhost:5432/db"
                    className="w-full rounded-md border border-zinc-700 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-zinc-400">Read-Only Query</label>
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="SELECT * FROM users WHERE active = true LIMIT 1000"
                    rows={3}
                    className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <button
                  onClick={handleSqlConnect}
                  disabled={!connectionString || !query || loading}
                  className="flex w-full items-center justify-center rounded-lg bg-emerald-500 py-2 text-sm font-medium text-zinc-950 transition-colors hover:bg-emerald-400 disabled:opacity-50"
                >
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Execute & Analyze'}
                </button>
              </div>
            )}
            
            {error && <p className="text-sm text-red-400">{error}</p>}
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-sm text-zinc-400">Select the target column to predict:</p>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500 focus:outline-none"
            >
              <option value="">-- Select Target --</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            
            <button
              onClick={handleConfirm}
              disabled={!targetColumn}
              className="w-full rounded-lg bg-emerald-500 py-2 text-sm font-medium text-zinc-950 transition-colors hover:bg-emerald-400 disabled:opacity-50"
            >
              Start MLPilot
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
