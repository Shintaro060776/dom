const express = require('express');
const axios = require('axios');
const app = express();
const port = 12000;

app.use(express.json());

app.post('/api/text2speech', async (req, res) => {
    try {
        const APIGATEWAY = 'https://n0agp0mzfb.execute-api.ap-northeast-1.amazonaws.com/prod/text2speech';

        const lambdaResponse = await axios.post(APIGATEWAY, {
            user_input: req.body.user_input
        });

        res.json(lambdaResponse);
    } catch (error) {
        console.error('Error calling Lambda:', error);
        res.status(500).send('Internal Server Error')
    }
});

app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});
