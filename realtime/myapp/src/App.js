import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const useTypewriter = (text, speed = 100) => {
  const [typed, setTyped] = useState('');
  const [blink, setBlink] = useState(true);

  useEffect(() => {
    let timeoutId;

    if (typed.length < text.length) {
      timeoutId = setTimeout(() => {
        setTyped((current) => current + text.charAt(current.length));
      }, speed);
    } else {
      const blinkInterval = setInterval(() => {
        setBlink((current) => !current);
      }, speed);

      return () => clearInterval(blinkInterval);
    }

    return () => clearTimeout(timeoutId);
  }, [text, typed, speed]);

  return { typed, blink };
};

function App() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const { typed, blink } = useTypewriter(response);

  const handleSubmit = async () => {
    try {
      const result = await axios.post('http://3.112.43.184/api/realtime', { input });
      setResponse(result.data.message);
    } catch (error) {
      console.error('Error fetching response:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Realtime ChatBot by Emotional Analysis/Open AI</h1>
        <a href="http://52.68.145.180/">トップページに戻る</a>
      </header>
      <div className="App-content">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="問い合わせ内容を入力してください"
        />
        <textarea
          value={typed}
          readOnly
          placeholder="AIからの回答が、ここに表示されます"
          className={`typewriter ${blink ? 'typewriter-blink' : ''}`}
        />
      </div>
      <button onClick={handleSubmit}>Inquiry for AI</button>
      <footer className="App-footer">
        <p>2023 Realtime ChatBot</p>
      </footer>
    </div>
  );
}

export default App;