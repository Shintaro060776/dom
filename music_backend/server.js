const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const port = 16000;

app.use(bodyParser.json());

const apiEndpoint1 = 'https://12zk6rip6j.execute-api.ap-northeast-1.amazonaws.com/prod/music1';
const apiEndpoint2 = 'https://poa6thi2jh.execute-api.ap-northeast-1.amazonaws.com/prod/music3';

app.post('/api/music-presigned', async (req, res) => {
    try {
        const response = await axios.post(`${apiEndpoint1}`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(error.response.status).send(error.response.data);
    }
});

app.get('/api/music', async (req, res) => {
    try {
        const response = await axios.get(`${apiEndpoint2}`);
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(error.response.status).send(error.response.data);
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

