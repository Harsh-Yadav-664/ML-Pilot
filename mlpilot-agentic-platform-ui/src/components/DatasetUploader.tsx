import React, { useState } from 'react';
import { Upload, X, Loader2 } from 'lucide-react';
import api, { runBaselineExperiment } from '../api';

interface DatasetUploaderProps {
  onSuccess: (datasetPath: string, targetColumn: string) => void;
  onCancel: () => void;
}

export function DatasetUploader({ onSuccess, onCancel }: DatasetUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
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
      // Direct raw axios call since it's a multipart form
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
            <div className="rounded-lg border-2 border-dashed border-zinc-800 p-8 text-center hover:bg-zinc-800/30">
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
