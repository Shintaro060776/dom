const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

app.post('/apt/gpt4', async (req, res) => {
    try {
        const response = await axios.post('https://i2i6rp7og4.execute-api.ap-northeast-1.amazonaws.com/prod/gpt4path', req.body);
        res.send(response.data);
    } catch (error) {
        console.error('Error making request to the API:', error);
        res.status(500).send({ error: 'Internal server error' });
    }
});

const PORT = 8000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});