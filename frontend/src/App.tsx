/**
 * AuthBrain AI Face Analysis Engine
 * App Root — Router Setup
 */

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConsentPage } from './pages/ConsentPage';
import { Dashboard }   from './pages/Dashboard';
import { useAuthStore, useAnalysisStore } from './store';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = useAuthStore(s => s.isAuthenticated);
  const consentGranted  = useAnalysisStore(s => s.consentGranted);

  if (!isAuthenticated || !consentGranted) {
    return <Navigate to="/" replace />;
  }
  return <>{children}</>;
};

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ConsentPage />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
