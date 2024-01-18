const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const port = 9000;

const apiGatewayUrl = 'ここにAPI GatewayのURLを記載';
const apiGatewayDynamoDBEndpoint = 'ここにAPI GatewayのURLを記載'

app.post('/api/upload', async (req, res) => {
    try {
        const { fileName } = req.body;

        const response = await axios.post(apiGatewayUrl, { fileName });

        res.json(response.data);
    } catch (error) {
        console.error('Error in /upload:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

app.get('/api/get', async (req, res) => {
    try {
        const { fileKey } = req.query;

        const response = await axios.get(`${apiGatewayDynamoDBEndpoint}?fileKey=${encodeURIComponent(fileKey)}`);

        res.json(response.data);
    } catch (error) {
        console.error('Error fetching summary:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});