import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, Polyline, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';
import './App.css';
import L from 'leaflet';

const App = () => {
  const [route, setRoute] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [distance, setDistance] = useState(0);
  const mapRef = useRef();

  const calculateDistance = useCallback(() => {
    let totalDistance = 0;
    for (let i = 1; i < route.length; i++) {
      const pointA = L.latLng(route[i - 1]);
      const pointB = L.latLng(route[i]);
      totalDistance += pointA.distanceTo(pointB);
    }
    setDistance(totalDistance / 1000);
  }, [route]);

  useEffect(() => {
    calculateDistance();
  }, [route, calculateDistance]);

  const saveRoute = async () => {
    try {
      const response = await axios.post('http://localhost:27000/api/save-route', {
        userId: 'user123',
        routeData: route,
      });
      console.log(response.data);
      alert('Route saved successfully');
    } catch (error) {
      console.error(error);
      alert('Failed to save route');
    }
  };

  const fetchRoutes = async () => {
    try {
      const response = await axios.get('http://localhost:27000/api/get-routes', {
        params: { userId: 'user123' },
      });
      setRoutes(response.data);
    } catch (error) {
      console.error(error);
      alert('Failed to fetch routes');
    }
  };

  const clearRoute = () => {
    setRoute([]);
    setDistance(0);
  };

  const AddMarkers = () => {
    useMapEvents({
      click(e) {
        setRoute((currentRoute) => [...currentRoute, [e.latlng.lat, e.latlng.lng]]);
      },
    });
    return null;
  };

  return (
    <div>
      <h1>Marathon Tracker</h1>
      <MapContainer center={[51.505, -0.09]} zoom={13} style={{ height: '400px', width: '100%' }} ref={mapRef}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <Polyline positions={route} color="blue" />
        <AddMarkers />
      </MapContainer>
      <div className="info-panel">
        <p>Total Distance: {distance.toFixed(2)} km</p>
      </div>
      <button onClick={saveRoute}>Save Route</button>
      <button onClick={fetchRoutes}>Fetch Routes</button>
      <button onClick={clearRoute}>Clear Route</button>
      <div>
        <h2>Saved Routes</h2>
        {routes.map((r, index) => (
          <div key={index} className="saved-route">
            <h3>Route {index + 1}</h3>
            <pre>{JSON.stringify(r.routeData, null, 2)}</pre>
          </div>
        ))}
      </div>
    </div>
  );
};

export default App;