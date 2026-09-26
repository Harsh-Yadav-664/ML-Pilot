import React from 'react';
import { useParams } from 'react-router-dom';

const ProjectPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Project</h1>
      <p className="text-gray-500">Project ID: <code className="bg-gray-100 px-2 py-0.5 rounded">{id}</code></p>
      <p className="mt-4 text-gray-400">Project detail view — Phase 1.</p>
    </div>
  );
};

export default ProjectPage;
