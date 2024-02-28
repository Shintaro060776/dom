import React from 'react';
import './App.css';
import axios from 'axios';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <a href="#home">Home</a>
      <a href="#archive">Archive</a>
      <a href="http://52.68.145.180/">Back To Top</a>
    </div>
  );
};

const Content = () => {
  const [inputText, setInputText] = React.useState('');
  const [generatedText, setGeneratedText] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);
  const [currentIndex, setCurrentIndex] = React.useState(0);
  const [displayedText, setDisplayedText] = React.useState('');

  React.useEffect(() => {
    if (isLoading) {
      setCurrentIndex(0);
      setDisplayedText('');
    } else if (currentIndex < generatedText.length) {
      const timeoutId = setTimeout(() => {
        setCurrentIndex(currentIndex + 1);
        setDisplayedText(generatedText.substring(0, currentIndex + 1));
      }, 150);

      return () => clearTimeout(timeoutId);
    }
  }, [generatedText, currentIndex, isLoading]);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleGenerateText = async () => {
    setIsLoading(true);
    setGeneratedText('テキストを生成しています...');

    try {
      const response = await axios.post('http://3.112.43.184/api/generate', {
        text: inputText,
      });

      setGeneratedText(response.data.generatedText);
    } catch (error) {
      console.error('Error fetching generated text:', error);
      setGeneratedText('テキストの生成に失敗しました...');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="content">
      <h1>Category Classification/Text Generation</h1>
      <h2>Where knowledge begins</h2>
      <input
        type="text"
        placeholder="生成する名言の、お題を入力してください"
        value={inputText}
        onChange={handleInputChange}
        disabled={isLoading}
      />
      <button onClick={handleGenerateText} className='round-button'>→</button>
      <p className='generated-text'>
        {displayedText}
        <span className='blinking-cursor'>|</span>
      </p>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <Sidebar />
      <Content />
    </div>
  );
}

export default App;