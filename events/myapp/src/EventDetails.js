import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';

const EventDetails = () => {
    const [event, setEvent] = useState(null);
    const { id } = useParams();

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

    if (!event) {
        return <div>Loading...</div>;
    }

    return (
        <div>
            <h2>{event.title}</h2>
            <p>{event.body}</p>
            <p>Date: {new Date(event.date).toLocaleDateString()}</p>
        </div>
    );
};

export default EventDetails;
