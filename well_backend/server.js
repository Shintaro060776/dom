const express = require('express');
const axios = require('axios');

const app = express();
const port = 5000;

app.use(express.json())

app.post('/api/generate', async (req, res) => {
    try {
        const response = await axios.post(
            'https://fai9gyqpg1.execute-api.ap-northeast-1.amazonaws.com/prod/generate',
            req.body
        );
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).send('Internal Server Error');
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

