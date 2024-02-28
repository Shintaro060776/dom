import React, { useState } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);

  const handleSendMessage = (userText, aiResponseText) => {
    const newUserMessage = { text: userText, sender: 'user' };
    const newAiMessage = { text: aiResponseText, sender: 'ai' };
    setMessages([...messages, newUserMessage, newAiMessage]);
  };

  return (
    <div className='app-container'>
      <MessageList messages={messages} />
      <MessageInput onSendMessage={handleSendMessage} />
    </div>
  );
}

export default App;