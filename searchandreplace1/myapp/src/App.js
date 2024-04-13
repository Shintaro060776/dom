import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css'; // CSSファイルをインポート

function App() {
  const [uploadedImage, setUpLoadedImage] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [searchPrompt, setSearchPrompt] = useState('');
  const fileInputRef = useRef(null);

  const getPresignedUrl = async (file) => {
    try {
      // APIエンドポイントのURLを更新してください
      const response = await axios.post('/api/searchandreplace', {
        fileName: file.name,
        prompt: prompt,
        searchPrompt: searchPrompt
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

  const handleFetchGeneratedImage = async () => {
    try {
      // APIエンドポイントのURLを更新してください
      const response = await axios.get('/api/latestsearchandreplace');
      setGeneratedImage(response.data.url);
    } catch (error) {
      console.error('Error fetching generated image', error);
    }
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <h1>Search and Replace by StabilityAI</h1>
        <a href='http://52.68.145.180/'>BackToTop</a>
      </header>
      <div className='content'>
        <div className='image-area'>
          <div className='image-container uploaded-image'>
            {uploadedImage ? <img src={uploadedImage} alt='Uploaded' /> : <div className="image-placeholder">Upload Image</div>}
          </div>
          <div className='image-container generated-image'>
            {generatedImage ? <img src={generatedImage} alt='Generated' /> : <div className="image-placeholder">Generated Image</div>}
          </div>
        </div>
        <div className='input-fields'>
          <input
            type='text'
            placeholder='Prompt'
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <input
            type="text"
            placeholder='Search Prompt'
            value={searchPrompt}
            onChange={(e) => setSearchPrompt(e.target.value)}
          />
        </div>
        <div className='buttons'>
          <button onClick={() => fileInputRef.current.click()}>Select Image</button>
          <input
            type='file'
            onChange={handleImageUpload}
            ref={fileInputRef}
            style={{ display: 'none' }}
          />
          <button onClick={handleFetchGeneratedImage}>Fetch Image</button>
        </div>
      </div>
    </div>
  );
}

export default App;