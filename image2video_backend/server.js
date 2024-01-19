const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

app.post('/api/image2video', async (req, res) => {
    try {
        const response = await axios.post(
            'https://kgqmlycmzc.execute-api.ap-northeast-1.amazonaws.com/prd/image',
            req.body
        );
        res.json(response.data);
    } catch (error) {
        if (error.response) {
            console.error("Error response from API:", error.response.data);
            res.status(error.response.status).json(error.response.data);
        } else if (error.request) {
            console.error("No response received:", error.request);
            res.status(500).json({ message: "No response received from API" });
        } else {
            console.error("Error setting up request:", error.message);
            res.status(500).json({ message: "Error sending request to API" });
        }
    }
});

const PORT = 10000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});