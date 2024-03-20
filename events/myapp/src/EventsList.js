import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';

const EventsList = () => {
    const [events, setEvents] = useState([]);

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const response = await axios.get('http://3.112.43.184/api/events');
                setEvents(response.data);
            } catch (error) {
                console.error("Error fetching events:", error);
            }
        };

        fetchEvents();
    }, []);

    return (
        <div>
            <h2>イベント一覧</h2>
            <Link to="/create-event">
                <button>Create New Event</button>
            </Link>
            <ul>
                {events.map(event => (
                    <li key={event.id}>
                        {event.title} - {new Date(event.date).toLocaleDateString()}
                        <Link to={`/events/${event.id}`}>
                            <button>Details</button>
                        </Link>
                        <Link to={`/events/edit/${event.id}`}>
                            <button>Edit</button>
                        </Link>
                        <Link to={`/events/delete/${event.id}`}>
                            <button>Delete</button>
                        </Link>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default EventsList;