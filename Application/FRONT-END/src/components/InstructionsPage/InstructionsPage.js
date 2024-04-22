import React from 'react';
import './InstructionsPage.css';
import Navbar from '../Navbar/Navbar';

function InstructionsPage() {
    return (
        <div className="instructions-container">
            <Navbar/>
            <div className="instructions-content">
                <h1>How to use AlzhiScan?</h1>
                <ul className="instructions-list">
                    <li className="list">AlzhiScan platform is capable of providing Alzheimer’s Disease stage detection and visual interpretations of the detection results.</li>
                    <li className="list">This platform is capable of identifying Alzheimer's Disease stages of 2D brain MRI images, uploading any other image will provide incorrect results.</li>
                    <li className="list">To get predictions for an MRI image, navigate to 'Predict' page from the above menu.</li>
                    <li className="list">In the 'Predict' page:</li>
                    <ul className="predict-steps">
                        <li className="list">1 - Select a 2D brain MRI image and upload it to AlzhiScan.</li>
                        <li className="list">2 - Click on ‘Preview’ button to view the uploaded MRI image.</li>
                        <li className="list">3 - If the uploaded image is correct, click on ‘Predict Stage’ button.</li>
                        <li className="list">4 - If you want to view the visual interpretations, click on the ‘View Visual Interpretations’ button.</li>
                    </ul>
                </ul>
            </div>
        </div>
    );
}

export default InstructionsPage;
