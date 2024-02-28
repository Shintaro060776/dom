import React, { useState } from 'react';
import axios from 'axios';
import './FileUpload.css';

const FileUpload = () => {
    const [file, setFile] = useState(null);
    const [summary, setSummary] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
        setSummary('');
    };

    const handleUpload = async () => {
        if (file) {
            setIsLoading(true);
            try {
                const response = await axios.post('http://3.112.43.184/api/upload', { fileName: file.name });
                const { url } = response.data;

                await axios.put(url, file, {
                    headers: {
                        'Content-Type': file.type,
                        'x-amz-acl': 'bucket-owner-full-control'
                    }
                });

                alert('File uploaded successfully');
            } catch (error) {
                console.error('Upload error:', error);
                if (error.response) {
                    console.error('Response:', error.response.data);
                    alert(`Error uploading file: ${error.response.status} ${error.response.statusText}`);
                } else {
                    alert('Error uploading file');
                }
            } finally {
                setIsLoading(false);
            }
        }
    };

    const fetchSummary = async () => {
        if (file) {
            setIsLoading(true);
            try {
                const summaryResponse = await axios.get('http://3.112.43.184/api/get', {
                    params: { fileKey: file.name }
                });

                setSummary(summaryResponse.data.summary);
            } catch (error) {
                console.error('Error fetching summary:', error);
                if (error.response) {
                    console.error('Response:', error.response.data);
                    alert(`Error fetching summary: ${error.response.status} ${error.response.statusText}`);
                } else {
                    alert('Error fetching summary');
                }
            } finally {
                setIsLoading(false);
            }
        } else {
            alert('Please upload a file first');
        }
    };

    return (
        <div>
            <header className='app-header'>
                <h1>Speech-to-Text/GPT4</h1>
                <nav>
                    <a href='http://52.68.145.180/'>トップページに戻る</a>
                </nav>
            </header>
            <div className="file-upload-container">
                <input type="file" className="file-input" onChange={handleFileChange} />
                <div className='file-upload-actions'>
                    <button className="upload-button" onClick={handleUpload}>Upload</button>
                    <button className="summary-button" onClick={fetchSummary} disabled={isLoading}>
                        {isLoading ? 'Loading...' : 'Show Summary'}
                    </button>
                </div>
                {summary && <div className="summary-container">
                    <h3>要約結果:</h3>
                    <p>{summary}</p>
                </div>}
            </div>
        </div>
    );
};

export default FileUpload;