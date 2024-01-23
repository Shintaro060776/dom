import './App.css';
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const uploadImageToS3 = async (presignedUrl, file) => {
    try {
      await axios.put(presignedUrl, file, {
        headers: {
          'Content-Type': file.type,
          'x-amz-acl': 'bucket-owner-full-control'
        }
      });
      return true;
    } catch (error) {
      console.error("Error uploading image to S3:", error);
      return false;
    }
  };


  const handleSubmit = async () => {
    if (!selectedFile) {
      alert("Please select a file first");
      return;
    }

    setLoading(true);
    try {
      const presignedResponse = await axios.post('http://3.112.43.184/api/image2video', {
        fileName: selectedFile.name
      });

      const presignedUrl = presignedResponse.data.url;

      const uploadSuccess = await uploadImageToS3(presignedUrl, selectedFile);

      if (!uploadSuccess) {
        setLoading(false);
        alert("Error uploading image to S3");
      }
    } catch (error) {
      console.error("Error getting presigned URL or uploading image:", error);
      setLoading(false);
      alert("Error during video generation process");
    }
  };

  const getLatestVideo = async () => {
    setLoading(true);

    try {
      const response = await axios.get('http://3.112.43.184/api/latest_video');
      setVideoUrl(response.data.videoUrl);
      setLoading(false);
    } catch (error) {
      console.error("Error retrieving latest video:", error);
      setLoading(false);
      alert("Error retrieving latest video");
    }
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <h1>Image-to-Video StabilityAI</h1>
        <nav>
          <a href='http://3.112.43.184/'>Home</a>
        </nav>
      </header>
      <div className='content'>
        <div className='image-preview'>
          {selectedFile && <img src={URL.createObjectURL(selectedFile)} alt='Selected' />}
        </div>
        <div className='video-preview'>
          {videoUrl && <video controls src={videoUrl}></video>}
        </div>
      </div>
      <div className='action'>
        <input type='file' onChange={handleFileSelect} />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? 'Generating...' : 'Generate Video'}
        </button>
        <button onClick={getLatestVideo} disabled={loading}>
          {loading ? 'Loading...' : 'Get Video'}
        </button>
      </div>
    </div>
  );
}

export default App;