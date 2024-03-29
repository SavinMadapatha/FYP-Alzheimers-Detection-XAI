import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './components/HomePage/HomePage';
import InstructionsPage from './components/InstructionsPage/InstructionsPage';
import PredictPage from './components/PredictPage/PredictPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/instructions" element={<InstructionsPage />} />
        <Route path="/predict" element={<PredictPage />} />
      </Routes>
    </Router>
  );
}

export default App;
