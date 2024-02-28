import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [audioUrl, setAudioUrl] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post('http://3.112.43.184/api/text2speech', { user_input: query });
      setAudioUrl(response.data.audio_url);
      const message = response.data.message;
      console.log("Received Message:", message);
    } catch (error) {
      console.error('Error', error);
      alert('An error occurred while processing your request');
    }
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <h1>Text2Speech</h1>
        <a href="http://3.112.43.184/">Home</a>
      </header>
      <main>
        <form onSubmit={handleSubmit}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder='Enter your query here...'
            rows="4"
          />
          {audioUrl && <audio src={audioUrl} controls />}
          <button type='submit'>Generate Speech</button>
        </form>
      </main>
    </div>
  );
}

export default App;

