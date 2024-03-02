import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState('');
  const [animatedImageUrl, setAnimatedImageUrl] = useState('');

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first');
      return;
    }

    const { data } = await axios.get('http://52.68.145.180/api/presigned-url');
    const { url } = data;

    await axios.put(url, selectedFile, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'x-amz-acl': 'bucket-owner-full-control',
      },
    });

    setUploadedImageUrl(URL.createObjectURL(selectedFile));
  };

  const handleFetchAnimatedImage = async () => {
    const { data } = await axios.get('http://52.68.145.180/api/latest-animated-image');
    setAnimatedImageUrl(data.imageUrl);
  };

  return (
    <div className='App'>
      <header className='header'>
        <h1>AILAB Converting to Animated Image</h1>
        <a href='http://52.68.145.180/'>Go to Top</a>
      </header>
      <input type='file' onChange={handleFileSelect} />
      <button className='handleUpload' onClick={handleUpload}>Upload Image</button>
      <button onClick={handleFetchAnimatedImage}>Fetch Animated Image</button>
      <div className='image-section'>
        <div className='image-container'>
          <h2>Uploaded Image</h2>
          {uploadedImageUrl && <img src={uploadedImageUrl} alt='uploaded' />}
        </div>
        <div className='image-container'>
          <h2>Animated Image</h2>
          {animatedImageUrl && <img src={animatedImageUrl} alt='Animated' />}
        </div>
      </div>
    </div>
  );
}

export default App;
