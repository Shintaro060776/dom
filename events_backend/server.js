const express = require('express');
const axios = require('axios');
const app = express();
const port = 18000;

app.use(express.json());

const ApigatewayEndpoint = 'https://0n7fdzi1b4.execute-api.ap-northeast-1.amazonaws.com/prod/event';

app.post('/api/events', async (req, res) => {
    try {
        const response = await axios.post(ApigatewayEndpoint, req.body);
        res.json(response.data);
    } catch (error) {
        handleError(error, res);
    }
});

app.get('/api/events/:id', async (req, res) => {
    try {
        const response = await axios.get(`${ApigatewayEndpoint}?id=${req.params.id}`);
        res.json(response.data);
    } catch (error) {
        handleError(error, res);
    }
});

app.put('/api/events/:id', async (req, res) => {
    try {
        const response = await axios.put(ApigatewayEndpoint, { ...req.body, id: req.params.id });
        res.json(response.data);
    } catch (error) {
        handleError(error, res);
    }
});

app.delete('/api/events/:id', async (req, res) => {
    try {
        const response = await axios.delete(ApigatewayEndpoint, { data: { id: req.params.id } });
        res.json(response.data);
    } catch (error) {
        handleError(error, res);
    }
});

function handleError(error, res) {
    if (error.response) {
        console.error('Error data:', error.response.data);
        console.error('Error status:', error.response.status);
        res.status(error.response.status).json(error.response.data);
    } else if (error.request) {
        console.error('No response:', error.request);
        res.status(500).json({ message: 'No response from server' });
    } else {
        console.error('Error', error.message);
        res.status(500).json({ message: error.message });
    }
}

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

