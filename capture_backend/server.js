const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const API_GATEWAY_URL_GET_PRESIGNED_URL = 'https://v9s8srsea4.execute-api.ap-northeast-1.amazonaws.com/prod/capture1'
const API_GATEWAY_URL_GET_IMAGES = 'https://kunycyzl60.execute-api.ap-northeast-1.amazonaws.com/prod/capture2';

app.post('/api/capture1-presigned-url', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_URL_GET_PRESIGNED_URL, req.body);
        res.status(200).send(response.data);
    } catch (error) {
        console.error('Error fetching presigned URL:', error);
        res.status(500).send({ error: 'Error fetching presigned URL' });
    }
});

app.get('/api/capture2-get-images', async (req, res) => {
    try {
        const response = await axios.get(API_GATEWAY_URL_GET_IMAGES);
        res.status(200).send(response.data);
    } catch (error) {
        console.error('Error fetching images:', error);
        res.status(500).send({ error: 'Error fetching images' });
    }
});

const PORT = 22000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

