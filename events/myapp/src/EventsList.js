import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';

const EventsList = () => {
    const [events, setEvents] = useState([]);

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const response = await axios.get('http://52.68.145.180/api/events');
                const formattedEvents = response.data.map(event => ({
                    ...event,
                    date: new Date(event.date.S),
                }));
                setEvents(formattedEvents);
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
                    <li key={event.id.S}>
                        {event.title.S} - {event.date.toLocaleDateString()}
                        <Link to={`/events/${event.id.S}`}>
                            <button>Details</button>
                        </Link>
                        <Link to={`/events/edit/${event.id.S}`}>
                            <button>Edit</button>
                        </Link>
                        <Link to={`/events/delete/${event.id.S}`}>
                            <button>Delete</button>
                        </Link>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default EventsList;