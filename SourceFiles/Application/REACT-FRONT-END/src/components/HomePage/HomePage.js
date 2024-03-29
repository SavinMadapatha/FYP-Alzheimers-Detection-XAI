import React from 'react';
import './HomePage.css';
import Navbar from '../Navbar/Navbar';
import { useNavigate } from 'react-router-dom';


function HomePage() {
    const navigate = useNavigate();

    const handleGetStartedBtnClick = () => {
        navigate('/instructions'); 
    };

    return (
        <div className="home-container">
            <Navbar/>
            <div className="content-container">
                <div className="text-content">
                    <p className="wordings">Unlocking Insights through Machine Learning.</p>
                    <p className="wordings">Accurate Alzheimer's Disease stage detection & visual interpretations.</p>
                    <p className="wordings">Ready to begin?</p>
                </div>
                <div className="divider"></div>
                <div className="button-container">
                    <button className="get-started-btn" onClick={handleGetStartedBtnClick}>Get Started</button>
                </div>
            </div>
        </div>
    );
}

export default HomePage;
