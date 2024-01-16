import React, { useState } from 'react';
import axios from 'axios';

function MessageInput({ onSendMessage }) {
    const [text, setText] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (text) {
            try {
                const response = await axios.post('http://3.112.43.184/api/gpt4', { message: text });
                const aiResponseText = response.data.response
                onSendMessage(text, aiResponseText);
            } catch (error) {
                console.error('Error sending message:', error);
            }
            setText('');
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input className='message-input' type="text" value={text} onChange={(e) => setText(e.target.value)} />
                <button type="submit">Send</button>
            </form>
            <div className='link-container'>
                <a href='http://3.112.43.184/' className='link-to-home'>トップページに戻る</a>
            </div>
        </div>
    );
}

export default MessageInput;