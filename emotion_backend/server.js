const express = require('express');
const axios = require('axios');
const app = express();

const port = process.env.PORT || 3000;

app.use(express.json());

const API_GATEWAY_URL = 'https://zt9ebxgz9c.execute-api.ap-northeast-1.amazonaws.com/prod/myresource';

app.post('/api/emotion', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_URL, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ message: 'Internal Server Error', error: error });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

