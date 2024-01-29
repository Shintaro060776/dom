const express = require('express');
const axios = require('axios');
const app = express();
const port = 12000;

app.use(express.json());

const apiGatewayUrl = 'https://n0agp0mzfb.execute-api.ap-northeast-1.amazonaws.com/prod/text2speech';

app.post('/api/text2speech', async (req, res) => {
    try {
        const lambdaResponse = await axios.post(apiGatewayUrl, { user_input: req.body.user_input });
        console.log("Received response from API Gateway:", lambdaResponse.data);
        res.json({
            message: lambdaResponse.data.message,
            audio_url: lambdaResponse.data.audio_url
        });
    } catch (error) {
        console.error('Error status:', error.response?.status);
        console.error('Error headers:', error.response?.headers);
        console.error('Error data:', error.response?.data);
        console.error('Error calling Lambda:', error);
        res.status(error.response?.status || 500).send(error.response?.data || 'Internal Server Error');
    }
});

app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});
