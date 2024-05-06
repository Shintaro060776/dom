const express = require('express');
const axios = require('axios');

const app = express();
const port = 23000;

const apiGatewayUrl = '';

app.use(express.json());

app.post('/api/smokefree', async (req, res) => {
    const userData = req.body;

    try {
        const response = await axios.post(apiGatewayUrl, userData);
        res.status(200).json({ message: 'Data sent successfully', data: response.data });
    } catch (error) {
        console.error('Error sending data to API Gateway:', error);
        res.status(500).json({ message: 'Failed to send data', error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});