import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';

const EventEdit = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [event, setEvent] = useState({
        title: '',
        date: '',
        body: ''
    });

    useEffect(() => {
        const fetchEventDetails = async () => {
            try {
                const response = await axios.get(`http://3.112.43.184/api/events/${id}`);
                setEvent(response.data);
            } catch (error) {
                console.error("Error fetching event details:", error);
            }
        };

        fetchEventDetails();
    }, [id]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setEvent(prevEvent => ({
            ...prevEvent,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await axios.put(`http://3.112.43.184/api/events/${id}`, event);
            navigate('/events');
        } catch (error) {
            console.error("Error updating event:", error);
        }
    };

    return (
        <div>
            <h2>Edit Event</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Title:</label>
                    <input type='text' name='title' value={event.title} onChange={handleChange} />
                </div>
                <div>
                    <label>Date:</label>
                    <input type='date' name='date' value={event.date} onChange={handleChange} />
                </div>
                <div>
                    <label>Body:</label>
                    <textarea name='body' value={event.body} onChange={handleChange}></textarea>
                </div>
                <button type='submit'>Update Event</button>
            </form>
        </div>
    );
};

export default EventEdit;