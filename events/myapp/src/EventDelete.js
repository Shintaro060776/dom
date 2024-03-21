import React from 'react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';

const EventDelete = () => {
    const { id } = useParams();
    const navigate = useNavigate();

    const handleDelete = async () => {
        if (window.confirm("Are you sure you wanna delete this event??")) {
            try {
                await axios.delete(`http://52.68.145.180/api/events/${id}`);
                navigate('/events');
            } catch (error) {
                console.error("Error deleting event:", error);
            }
        }
    };

    return (
        <div>
            <h2>Delete Event</h2>
            <p>Are you sure you wanna delete this event??</p>
            <button onClick={handleDelete}>Delete button</button>
        </div>
    );
};

export default EventDelete;
