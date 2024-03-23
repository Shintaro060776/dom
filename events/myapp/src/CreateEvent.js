import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { v4 as uuid4 } from 'uuid';

const CreateEvent = () => {
    const [title, setTitle] = useState('');
    const [date, setDate] = useState('');
    const [body, setBody] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        const id = uuid4();
        e.preventDefault();
        try {
            const response = await axios.post('http://52.68.145.180/api/events', { id, title, date, body });
            console.log(response.data);
            navigate('/events');
        } catch (error) {
            console.error("Error creating event:", error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <label>Title:
                <input type='text' value={title} onChange={(e) => setTitle(e.target.value)} />
            </label>
            <label>Date:
                <input type='date' value={date} onChange={(e) => setDate(e.target.value)} />
            </label>
            <label>Body:
                <textarea value={body} onChange={(e) => setBody(e.target.value)}></textarea>
            </label>
            <button type='submit'>Create Event</button>
        </form>
    );
};

export default CreateEvent;
