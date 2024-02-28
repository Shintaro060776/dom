import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [imageUrl, setImageUrl] = useState('');
  const [inputText, setInputText] = useState('');

  const fetchImage = async () => {
    try {
      const response = await axios.post('http://3.112.43.184/api/dalle', {
        text: inputText
      });
      setImageUrl(response.data.imageUrl);
    } catch (error) {
      console.error("Error fetching image: ", error);
    }
  };

  return (
    <div className='app'>
      <header className='header'>
        <h2>Generate Image by Dall・E</h2>
        <a href='http://52.68.145.180/'>BackToTopPage</a>
      </header>
      <div className='body'>
        <div className='input-container'>
          <input
            type='text'
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder='Please write what you wanna generate'
          />
          <button onClick={fetchImage} className='fetch-button'>Generate Image</button>
          <div className='image-container'>
            {imageUrl && <img src={imageUrl} alt='Generated' />}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
