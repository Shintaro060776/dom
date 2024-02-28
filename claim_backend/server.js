const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const API_GATEWAY_URL = 'https://u59sumxsdl.execute-api.ap-northeast-1.amazonaws.com/prod/claim_handler';

app.post('/api/claim', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_URL, req.body);
        res.json(response.data);
    } catch (error) {
        console.log("Error forwarding request:", error);
        res.status(500).send("Error forwarding request to API Gateway");
    }
});

const PORT = 6000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});