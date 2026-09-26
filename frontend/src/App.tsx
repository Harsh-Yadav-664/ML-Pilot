import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import ProjectPage from './pages/ProjectPage';
import DataPage from './pages/DataPage';
import ExperimentsPage from './pages/ExperimentsPage';
import ModelPage from './pages/ModelPage';
import ReportsPage from './pages/ReportsPage';

const navLinks = [
  { to: '/', label: 'Dashboard', exact: true },
  { to: '/data', label: 'Data' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/models', label: 'Models' },
  { to: '/reports', label: 'Reports' },
];

const App: React.FC = () => (
  <BrowserRouter>
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <nav className="w-56 bg-indigo-900 text-white flex flex-col py-8 px-4 gap-2 shrink-0">
        <div className="mb-8">
          <span className="text-2xl font-extrabold tracking-tight">ML<span className="text-indigo-300">Pilot</span></span>
          <div className="text-xs text-indigo-400 mt-1">Phase 0</div>
        </div>
        {navLinks.map(link => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.exact}
            className={({ isActive }) =>
              `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive ? 'bg-indigo-700 text-white' : 'text-indigo-200 hover:bg-indigo-800'
              }`
            }
          >
            {link.label}
          </NavLink>
        ))}
      </nav>
      {/* Main */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/projects/:id" element={<ProjectPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/models" element={<ModelPage />} />
          <Route path="/reports" element={<ReportsPage />} />
        </Routes>
      </main>
    </div>
  </BrowserRouter>
);

export default App;
