import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import EventsList from './EventsList';
import EventDetails from './EventDetails';
import EventDelete from './EventDelete';
import EventEdit from './EventEdit';
import CreateEvent from './CreateEvent';
import './App.css';
import { Link } from 'react-router-dom';

function App() {
  return (
    <Router>
      <div>
        <header className='header'>
          <h2>Event Management</h2>
          <nav>
            <Link to="http://52.68.145.180/">Back to Home</Link>
          </nav>
        </header>
        <main className='main-content'>
          <Routes>
            <Route path='/events' element={<EventsList />} />
            <Route path='/events/:id' element={<EventDetails />} />
            <Route path='/events/edit/:id' element={<EventEdit />} />
            <Route path='/events/delete/:id' element={<EventDelete />} />
            <Route path='/create-event' element={<CreateEvent />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
