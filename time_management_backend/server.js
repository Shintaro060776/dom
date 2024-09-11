const express = require('express');
const axios = require('axios');

const app = express();
const port = 29000;

app.use(express.json());

app.post('/api/time_management', async (req, res) => {
    const { url } = req.body;

    try {
        const response = await axios.get(url);
        res.status(200).json({
            data: response.data,
            message: 'Successfully fetched data from external API'
        });
    } catch (error) {
        res.status(500).json({
            message: 'Error occurred while calling external API',
            error: error.message
        });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});