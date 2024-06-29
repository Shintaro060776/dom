import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [image, setImage] = useState('');

  const handleGenerate = async () => {
    try {
      const response = await axios.post('http://52.68.145.180/api/prompt-gen', { prompt });
      const imageUrl = response.data.imageUrl;
      alert(`Image Generated! Image URL: ${imageUrl}`);
    } catch (error) {
      console.error('Error generating image:', error);
      alert('Failed to generate image');
    }
  };

  const handleGetImage = async () => {
    try {
      const response = await axios.get('http://52.68.145.180/api/get-prompt-gen', { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      setImage(url);
    } catch (error) {
      console.error('Error getting image:', error);
      alert('Failed to get image');
    }
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <h1>Image Generator by StabilityAI Latest Model</h1>
        <a href="http://52.68.145.180/">トップページに戻る</a>
      </header>
      <div className='input-section'>
        <input
          type='text'
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder='Enter Prompt'
        />
        <div className='button-section'>
          <button onClick={handleGenerate}>Generate Image</button>
          <button onClick={handleGetImage}>Get Image</button>
        </div>
      </div>
      {image && <img src={image} alt='Generated' className='generated-image' />}
    </div>
  );
}

export default App;