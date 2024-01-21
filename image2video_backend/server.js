const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const port = 10000;

app.use(bodyParser.json());

app.post('/api/image2video', async (req, res) => {
    try {
        const response = await axios.post(
            'https://qse55vmn5m.execute-api.ap-northeast-1.amazonaws.com/prod/stabilityai',
            req.body
        );
        res.json(response.data);
    } catch (error) {
        console.error('Error on forwarding request to API Gateway:', error.message);
        res.status(500).json({ 'axios': 'Internal Server Error' });
    }
});

app.get('/api/check_video_status/:generationId', async (req, res) => {
    try {
        const generationId = req.params.generationId;
        const response = await axios.get(`API_GATEWAY_ENDPOINT/check_video_status/${generationId}`);
        res.json(response.data);
    } catch (error) {
        console.error("Error forwarding request to API Gateway:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});