import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [uploadedImage, setUpLoadedImage] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [additionalPrompt, setAdditionalPrompt] = useState('');
  const [mainPrompt, setMainPrompt] = useState('');
  const fileInputRef = useRef(null);

  const getPresignedUrl = async (file) => {
    try {
      const response = await axios.post('http://3.112.43.184/api/image2image', {
        fileName: file.name,
        prompt: mainPrompt,
        additionalPrompt: additionalPrompt
      });
      return response.data.url;
    } catch (error) {
      console.error('Error getting presigned URL', error);
    }
  };

  const uploadImageToS3 = async (file) => {
    const presignedUrl = await getPresignedUrl(file);
    if (!presignedUrl) return;

    try {
      await axios.put(presignedUrl, file, {
        headers: {
          'Content-Type': file.type,
          'x-amz-acl': 'bucket-owner-full-control'
        }
      });
      setUpLoadedImage(URL.createObjectURL(file));
    } catch (error) {
      console.error('Error uploading to S3', error);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadImageToS3(file);
    }
  };

  const handleGetNewImage = async () => {
    try {
      const response = await axios.get('http://3.112.43.184/api/latest_image');
      setGeneratedImage(response.data.url);
    } catch (error) {
      console.error('Error getting new image', error);
    }
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <nav>
          <a href='http://52.68.145.180/'>Home</a>
        </nav>
      </header>

      <div className='content'>
        <div className='image-area'>
          <div className='uploaded-image'>
            {uploadedImage ? <img src={uploadedImage} alt='Uploaded' /> : <div className="image-placeholder">Upload Image</div>}
          </div>
          <div className='generated-image'>
            {generatedImage ? <img src={generatedImage} alt='Generated' /> : <div className="image-placeholder">Generated Image</div>}
          </div>
        </div>
        <div className='input-fields'>
          <input
            type='text'
            placeholder='Main Prompt'
            value={mainPrompt}
            onChange={(e) => setMainPrompt(e.target.value)}
          />
          <input
            type="text"
            placeholder='Additional Prompt'
            value={additionalPrompt}
            onChange={(e) => setAdditionalPrompt(e.target.value)}
          />
        </div>
        <div className='buttons'>
          <input
            type='file'
            onChange={handleImageUpload}
            ref={fileInputRef}
            style={{ display: 'none' }}
          />
          <button onClick={() => fileInputRef.current.click()}>Select Image</button>
          <button onClick={handleGetNewImage}>New Image Get</button>
        </div>
      </div>
    </div>
  );
}

export default App;