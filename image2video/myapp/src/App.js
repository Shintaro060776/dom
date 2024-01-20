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

  const handleSubmit = async () => {
    if (!selectedFile) {
      alert("Please select a file first");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://localhost:10000/api/image2video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setVideoUrl(response.data.videoUrl);
      setLoading(false);
    } catch (error) {
      console.error("Error uploading image:", error);
      setLoading(false);
      alert("Error uploading image");
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
      </div>
    </div>
  );
}

export default App;