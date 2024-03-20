import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const CreateEvent = () => {
    const [title, setTitle] = useState('');
    const [date, setDate] = useState('');
    const [body, setBody] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://3.112.43.184/api/events', { title, date, body });
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
