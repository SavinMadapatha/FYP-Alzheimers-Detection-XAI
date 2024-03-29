import React, { useState } from 'react';
import './PredictPage.css';
import Navbar from '../Navbar/Navbar';
import FolderImage from '/Users/savin/Desktop/FYP/final_chapters/Application/alzhiscan-web-app/src/assets/images/folderImage.png';

function PredictPage() {
    const backendBaseUrl = 'http://127.0.0.1:5000';
    const [step, setStep] = useState(1);
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const [uploadedImage, setUploadedImage] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [isUploaded, setIsUploaded] = useState(false);
    const [fileType, setFileType] = useState('');
    const [fileSize, setFileSize] = useState(0);
    const [isLoadingXAI, setIsLoadingXAI] = useState(false);

    // Function to handle file selection
    const handleFileSelect = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setFileName(selectedFile.name);
            setFileType(selectedFile.type);
            setFileSize(selectedFile.size);
            setUploadProgress(0);
            setIsUploaded(false);
        }
    };

    // Simulate the upload process
    const simulateUpload = () => {
        setUploadProgress(0); 
        const interval = setInterval(() => {
            setUploadProgress(oldProgress => {
                const newProgress = Math.min(oldProgress + 10, 100);
                if (newProgress === 100) {
                    clearInterval(interval);
                    setIsUploaded(true); 
                }
                return newProgress;
            });
        }, 100);
    };

    // Function to handle file upload
    const handleUpload = () => {
        if (file) {
            simulateUpload();
        }
    };

    // Function to handle previewing the MRI image
    const handlePreview = () => {
        const imageUrl = URL.createObjectURL(file);
        setUploadedImage(imageUrl);
        setStep(2);
    };    

    // Function to handle going to the next step
    const nextStep = () => {
        if (step < 3) {
            setStep(step + 1);
        }
    };

    // Function to handle going to the previous step
    const prevStep = () => {
        if (step > 1) {
            setStep(step - 1);
        }
    };

    // variables to hold the model's prediction results
    const [predictionResult, setPredictionResult] = useState({
        prediction: '',
        confidence: '',
        description: '',
        gradCamUrls: [],
    });

    // Function to handle the predict stage button click
    const handlePredictStageClick = async () => {
        if (!file) return;
    
        const formData = new FormData();
        formData.append('imagefile', file);
    
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            console.log("Server response:", data);
    
            if (response.ok) {
                // Store prediction, confidence, and description in the state
                // Do not attempt to handle XAI interpretation images here
                setPredictionResult({
                    prediction: data[0],
                    confidence: data[1],
                    description: data[2],
                    xai_image_urls: [], // Initialize as an empty array
                });
                setStep(3); // Navigate to the Detection Result step
            } else {
                console.error('Failed to fetch prediction');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };


    const handleVisualInterpretationsClick = async () => {
        // Navigate immediately to the Visual Interpretations step
        setStep(4);
        setIsLoadingXAI(true); // Start showing the loading indicator
    
        if (!file) {
            console.error('No file selected');
            setIsLoadingXAI(false); // Ensure to stop the loading indicator
            return;
        }
    
        const formData = new FormData();
        formData.append('imagefile', file);
    
        try {
            const response = await fetch(`${backendBaseUrl}/generate_interpretations`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
    
            if (response.ok) {
                // Update the state with URLs for the XAI interpretation images
                setPredictionResult(prevState => ({
                    ...prevState,
                    xai_image_urls: Array.isArray(data) ? data : [data],
                }));
            } else {
                console.error('Failed to fetch XAI interpretations');
            }
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setIsLoadingXAI(false); // Stop showing the loading indicator regardless of outcome
        }
    };
    
    // Renders the content based on the step
    const renderStepContent = () => {
        
        switch (step) {
            case 1:
                return (
                    <>
                        <div className="file-upload-container">
                            <div className="file-input-container">
                                <input type="file" onChange={handleFileSelect} className="file-input" />
                            </div>
                            <button className="btn" onClick={handleUpload} disabled={!file}>Upload</button>
                        </div>
                        {file && ( 
                            <div className="upload-info-container">
                                <img src={FolderImage} alt="Folder" className="folder-image" />
                                <div className="upload-info">
                                    <span className="file-name">{fileName}</span>
                                    <progress value={uploadProgress} max="100"></progress>
                                    <span className="upload-percentage">{uploadProgress}%</span>
                                </div>
                            </div>
                        )}
                        <div className="step-actions">
                                {uploadProgress === 100 && <button className="btn btn-preview" onClick={handlePreview}>Preview</button>}
                        </div>
                    </>
                );
            case 2:
                return (
                    <>
                        <div className="preview-container">
                            <img src={uploadedImage} alt="Uploaded MRI" className="preview-image" />
                            <div className="file-details">
                                <p><strong>File name:</strong> {fileName}</p>
                                <p><strong>File type:</strong> {fileType}</p>
                                <p><strong>File size:</strong> {fileSize.toLocaleString()} bytes</p>
                            </div>
                        </div>
                        <div className="step-actions">
                            <button className="btn btn-back" onClick={() => setStep(1)}>Go Back</button>
                            <button className="btn btn-predict" onClick={handlePredictStageClick}>Predict Stage</button>
                        </div>
                    </>
                );
            
            case 3:
                return (
                    <>
                        <div className="result-container">
                            <div className="result-top-section"> {/* New wrapper div */}
                                <div className="result-details">
                                    <div className="detail-item"><span className="detail-label">File name</span><span>: {fileName}</span></div>
                                    <div className="detail-item"><span className="detail-label">Detected Alzheimer's Stage</span><span>: {predictionResult.prediction}</span></div>
                                    <div className="detail-item"><span className="detail-label">Confidence</span><span>: {predictionResult.confidence}</span></div>
                                    <div className="detail-item"><span className="detail-label">Description</span><span>: {predictionResult.description}</span></div>
                                </div>
                                <img src={uploadedImage} alt="Uploaded MRI" className="result-image" />
                            </div>

                            <div className="confidence-bars">
                                <h3 className="confidence-bar-heading">Results Summary</h3>
                                {["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"].map((stage) => (
                                    <div className="confidence-bar" key={stage}>
                                    <div className="confidence-bar-content">
                                        <p>{stage} Stage</p>
                                        <p className="confidence-percentage">{predictionResult.prediction === stage ? predictionResult.confidence : "0%"}</p>
                                    </div>
                                    <progress value={predictionResult.prediction === stage ? predictionResult.confidence.replace('%', '') : "0"} max="100"></progress>
                                    </div>
                                ))}
                            </div>
                        </div>
                        
                        <div className="step-actions">
                            <button className="btn btn-back" onClick={() => setStep(1)}>New Scan</button>
                            <button className="btn btn-interpretations" onClick={handleVisualInterpretationsClick}>Visual Interpretations</button>
                        </div>
                    </>
                );
            case 4:
                // Assuming gradCamUrls is an array where even indexes are for Model 1 and odd indexes are for Model 2
                const model1Urls = predictionResult.xai_image_urls.filter((_, index) => index % 2 === 0);
                const model2Urls = predictionResult.xai_image_urls.filter((_, index) => index % 2 !== 0);
                
                return (
                    <>
                        <div className="interpretations-heading-container">
                            <div className="model-interpretations">
                                {isLoadingXAI ? (
                                    <div className="loading-container">
                                        <div className="spinner"></div>
                                        <p>Loading interpretations...</p>
                                    </div>
                                ) : (
                                    <>
                                        <div className="interpretations-column" style={{ marginRight: '10px' }}> {/* First model interpretations */}
                                            <h2>Model 1 Interpretations</h2>
                                            {model1Urls.map((url, index) => (
                                                <div className="interpretation-div" key={`model-1-${index}`}>
                                                    <img src={`${backendBaseUrl}${url}`} alt={`Model 1 Interpretation ${index + 1}`} className="interpretation-image" />
                                                </div>
                                            ))}
                                        </div>
                                        <div className="interpretations-column"> {/* Second model interpretations */}
                                            <h2>Model 2 Interpretations</h2>
                                            {model2Urls.map((url, index) => (
                                                <div className="interpretation-div" key={`model-2-${index}`}>
                                                    <img src={`${backendBaseUrl}${url}`} alt={`Model 2 Interpretation ${index + 1}`} className="interpretation-image" />
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                        <div className="step-actions">
                            <button className="btn btn-back" onClick={() => setStep(1)}>New Scan</button>
                        </div>
                    </>
                );
                  
            default:
                return null;
        }
    };

    return (
        <div className="predict-container">
            <Navbar />
            <div className="steps-container">
                <div className="step-indicator-container">
                    <div className={`step ${step === 1 ? 'active' : ''}`}>1 Upload MRI</div>
                    <div className={`step ${step === 2 ? 'active' : ''}`}>2 Preview MRI</div>
                    <div className={`step ${step === 3 ? 'active' : ''}`}>3 Detection Result</div>
                    <div className={`step ${step === 4 ? 'active' : ''}`}>4 Visual Interpretations</div>
                </div>
                <div className="step-content">
                    {renderStepContent()}
                </div>
            </div>
        </div>
    );
}

export default PredictPage;