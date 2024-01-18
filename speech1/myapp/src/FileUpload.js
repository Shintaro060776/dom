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
                        'Content-Type': file.type
                    }
                });

                alert('File uploaded successfully');
                setIsLoading(false);
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
                const summaryResponse = await axios.get('http://3.112.43.184/api/get', { params: { fileKey: file.name } });
                setSummary(summaryResponse.data.summary);
                setIsLoading(false);
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
        };

        return (
            <div className="file-upload-container">
                <a href='http://3.112.43.184/'>トップページに戻る</a>
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
        );
    };
};

export default FileUpload;