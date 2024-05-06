import './App.css';
import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [smokeFreeDays, setSmokeFreeDays] = useState(0);
  const [cigarettesNotSmoked, setCigarettesNotSmoked] = useState(0);
  const [moneySaved, setMoneySaved] = useState(0);
  const [aiResponse, setAiResponse] = useState('');
  const [displayedText, setDisplayedText] = useState('');
  const [cursor, setCursor] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setCursor(prev => !prev);
    }, 500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < aiResponse.length) {
        setDisplayedText(prev => prev + aiResponse[index]);
        index++;
      } else {
        clearInterval(interval);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [aiResponse]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const userData = {
      smoke_free_days: smokeFreeDays,
      cigarettes_not_smoked: cigarettesNotSmoked,
      money_saved: moneySaved
    };

    try {
      const response = await axios.post('/api/smokefree', userData);
      console.log('Response:', response.data);
      setAiResponse(response.data.data.ai_response);
      setDisplayedText('');
      alert('データが正常に送信されました');
    } catch (error) {
      alert('データの送信に失敗しました');
      console.error('Send data error:', error);
    }
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <a href="http://52.68.145.180/" className="Home-link">ホームへ戻る</a>
        <h1>禁煙進捗アプリ</h1>
        <form onSubmit={handleSubmit}>
          <label htmlFor='smokeFreeDays'>禁煙日数</label>
          <input
            type='number'
            value={smokeFreeDays}
            onChange={e => setSmokeFreeDays(e.target.value)}
            placeholder='禁煙日数'
          />
          <label htmlFor="cigarettesNotSmoked">未喫煙のタバコの本数:</label>
          <input
            type='number'
            value={cigarettesNotSmoked}
            onChange={e => setCigarettesNotSmoked(e.target.value)}
            placeholder='未喫煙のタバコの本数'
          />
          <label htmlFor="moneySaved">節約した金額:</label>
          <input
            type='number'
            value={moneySaved}
            onChange={e => setMoneySaved(e.target.value)}
            placeholder='節約した金額'
          />
          <button type='submit'>データ送信</button>
        </form>
        <div className='aiResponse'>
          <h2>AIの応答:</h2>
          <p>{displayedText}<span className={cursor ? 'cursor' : ''}>|</span></p>
        </div>
      </header>
    </div>
  );
}

export default App;