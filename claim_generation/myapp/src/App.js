import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'


function App() {
  const [userInput, setUserInput] = useState('');
  const [responseText, setResponseText] = useState('');
  const [typedText, setTypedText] = useState('');
  const [blink, setBlink] = useState('');

  useEffect(() => {
    if (responseText.length > 0 && typedText.length < responseText.length) {
      const timeoutId = setTimeout(() => {
        setTypedText(responseText.slice(0, typedText.length + 1));
      }, 100);
      return () => clearTimeout(timeoutId);
    }
  }, [responseText, typedText]);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setBlink((blink) => !blink);
    }, 530);
    return () => clearInterval(intervalId);
  }, []);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://3.112.43.184/api/claim', { user_input: userInput });
      setResponseText(response.data.final_text);
    } catch (error) {
      console.log('Error sending data to the server', error);
    }
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <h1>Dealing with Claim</h1>
        <nav>
          <a href='/'>Home</a>
          <a href='http://52.68.145.180/'>BackToTopPage</a>
        </nav>
      </header>

      <div className='container'>
        <div className='input-section'>
          <textarea
            placeholder='Enter your claim here...'
            value={userInput}
            onChange={handleInputChange}
          />

        </div>
        <div className='response-section'>
          <div
            className='fake-textarea'
            aria-readonly="true"
          >
            {typedText}
            <span className={blink ? 'cursor blink' : 'cursor'}>|</span>
          </div>
        </div>
      </div>
      <button onClick={handleSubmit}>Submit</button>
    </div>
  );
}

export default App;