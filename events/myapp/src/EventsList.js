import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';

const EventsList = () => {
    const [events, setEvents] = useState([]);

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const response = await axios.get('http://52.68.145.180/api/events');
                // response.dataは既に期待される形式のオブジェクトの配列であるため、直接利用する
                const formattedEvents = response.data.map(event => {
                    // dateプロパティをDateオブジェクトに変換
                    // eventオブジェクトの構造に合わせて適切にアクセスしてください
                    const eventDate = new Date(event.date); // event.date.Sから変更
                    return {
                        ...event,
                        date: eventDate,
                    };
                });
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
                    <li key={event.id}> {/* event.id.Sから変更 */}
                        {event.title} - {event.date.toLocaleDateString()} {/* event.title.Sから変更 */}
                        <Link to={`/events/${event.id}`}> {/* event.id.Sから変更 */}
                            <button>Details</button>
                        </Link>
                        <Link to={`/events/edit/${event.id}`}> {/* event.id.Sから変更 */}
                            <button>Edit</button>
                        </Link>
                        <Link to={`/events/delete/${event.id}`}> {/* event.id.Sから変更 */}
                            <button>Delete</button>
                        </Link>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default EventsList;