import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [image, setImage] = useState('');

  const generateImage = async () => {
    try {
      const response = await axios.post('http://52.68.145.180/api/text2image', { text });
      const imageUrl = response.data.imageUrls[0];
      setImage(imageUrl);
    } catch (error) {
      console.error('Error generating image:', error);
    }
  };

  return (
    <div className='app'>
      <header className='app-header'>
        <h2>StabilityAI Image Generation</h2>
        <a href='http://52.68.145.180'>トップページに戻る</a>
      </header>
      <div className='content'>
        <input
          type='text'
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder='Enter text for image generation'
          className='text-input'
        />
        <button onClick={generateImage} className='generate-button'>
          Generate Image
        </button>
        <div className='image-container'>
          {image && <img src={image} alt='Generated' className='generated-image' />}
        </div>
      </div>
    </div>
  );
}

export default App;
