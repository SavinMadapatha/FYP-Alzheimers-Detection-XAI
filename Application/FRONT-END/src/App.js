
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LoginPage from './components/LoginPage/LoginPage';
import HomePage from './components/HomePage/HomePage';
import InstructionsPage from './components/InstructionsPage/InstructionsPage';
import PredictPage from './components/PredictPage/PredictPage';
import PrivateRoute from './components/LoginPage/PrivateRoute';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/home" element={<PrivateRoute><HomePage /></PrivateRoute>} />
        <Route path="/instructions" element={<PrivateRoute><InstructionsPage /></PrivateRoute>} />
        <Route path="/predict" element={<PrivateRoute><PredictPage /></PrivateRoute>} />
      </Routes>
    </Router>
  );
}

export default App;
