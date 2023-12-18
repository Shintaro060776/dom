import React, { useState } from "react";
import axios from 'axios';
import './App.css';

const App = () => {
  const [inputText, setInputText] = useState('');
  const [response, setResponse] = useState(null);
  const [emotionLevel, setEmotionLevel] = useState(3);

  const getEmotionSentence = (level) => {
    switch (level) {
      case '1':
        return "幸せなようです。";
      case '2':
        return "満足しているようです。";
      case '3':
        return "感情的に普通な状態のようです。";
      case '4':
        return "イライラされているようです。";
      case '5':
        return "怒っているようです";
      default:
        return '';
    }
  };

  const handleInquiry = async () => {
    const emotionSentence = getEmotionSentence(emotionLevel);
    const fullText = `${inputText} ${emotionSentence}`;

    try {
      const response = await axios.post('http://3.112.43.184/api/emotion', { text: fullText });
      setResponse(response.data);
    } catch (error) {
      console.error('APIエラー:', error);
    }
  };

  return (
    <div>
      <Header />
      <div id='container'>
        <div id='inquiry-section'>
          <textarea
            placeholder='Enter your inquiry here.....'
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          <EmotionSlider
            emotionLevel={emotionLevel}
            onEmotionChange={setEmotionLevel}
          />
          <button onClick={handleInquiry}>Inquiry</button>
        </div>
        <ResponseDisplay response={response} />
      </div>
      <Footer />
    </div>
  );
};

const EmotionSlider = ({ emotionLevel, onEmotionChange }) => {
  const handleSliderChange = (event) => {
    onEmotionChange(event.target.value);
  };

  return (
    <div className="emotion-slider-container">
      <input
        type="range"
        min="1"
        max="5"
        value={emotionLevel}
        onChange={handleSliderChange}
      />
      <div style={{ color: 'white' }}>Emotion level : {getEmotionText(emotionLevel)}</div>
    </div>
  );
};

const getEmotionText = (level) => {
  switch (level) {
    case '1':
      return "Happy😆";
    case '2':
      return "Content😀";
    case '3':
      return "Neutral😐";
    case '4':
      return "Annoyed😟";
    case '5':
      return "Angry😡";
    default:
      return 'Neutral😐';
  }
};

const Header = () => (
  <header style={{ padding: '10px', backgroundColor: '#333', color: 'white', textAlign: 'center' }}>
    <h1>Inquiry by Emotion Analysis</h1>
  </header>
);

const Footer = () => (
  <footer style={{ padding: '10px', backgroundColor: '#333', color: 'white', textAlign: 'center' }}>
    <p>© 2022 Emotion Analysis Inquiry. All rights reserved.</p>
  </footer>
);

const ResponseDisplay = ({ response }) => (
  <div id='response-section'>
    {response && <div>{response}</div>}
  </div>
);

export default App;