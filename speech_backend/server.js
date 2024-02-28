const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const port = 9000;

const apiGatewayUrl = 'https://wjzbog50ma.execute-api.ap-northeast-1.amazonaws.com/prod/speech2';
const apiGatewayDynamoDBEndpoint = 'https://td3tuwkskg.execute-api.ap-northeast-1.amazonaws.com/prod/speech3'

app.post('/api/upload', async (req, res) => {
    try {
        const { fileName } = req.body;

        const response = await axios.post(apiGatewayUrl, { fileName });

        res.json(response.data);
    } catch (error) {
        console.error('Error in /upload:', error);
        if (error.response) {
            console.error('Response:', error.response.data);
            res.status(error.response.status).json({ message: error.response.data });
        } else {
            res.status(500).json({ message: 'Internal server error' });
        }
    }
});

app.get('/api/get', async (req, res) => {
    try {
        const { fileKey } = req.query;

        const response = await axios.get(`${apiGatewayDynamoDBEndpoint}?fileKey=${encodeURIComponent(fileKey)}`);

        res.json(response.data);
    } catch (error) {
        console.error('Error fetching summary:', error);
        if (error.response) {
            console.error('Response:', error.response.data);
            res.status(error.response.status).json({ message: error.response.data });
        } else {
            res.status(500).json({ message: 'Internal server error' });
        }
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});