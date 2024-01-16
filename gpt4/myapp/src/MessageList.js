import React from 'react';

function MessageList({ messages }) {
    return (
        <div>
            {messages.map((message, index) => (
                <div key={index} className={message.sender === 'user' ? 'user-message' : 'ai-message'}>
                    {message.text}
                </div>
            ))}
        </div>
    );
}

export default MessageList;