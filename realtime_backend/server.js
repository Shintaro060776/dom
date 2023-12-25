const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

app.post('/api/realtime', async (req, res) => {
    try {
        const { user_input } = req.body;

        const lambdaResponse = await axios.post(
            'https://6xc8jru0di.execute-api.ap-northeast-1.amazonaws.com/prod/realtime',
            { user_input }
        );

        res.status(200).json(lambdaResponse.data);
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

