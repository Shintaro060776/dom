const express = require('express');
const axios = require('axios');
const app = express();
const port = 13000;

app.use(express.json());

app.post('/api/text2image', async (req, res) => {
    try {
        const { data } = await axios.post('https://efyrqso33g.execute-api.ap-northeast-1.amazonaws.com/prod/text2image', req.body, {
            Headers: {
                'Content-Type': 'application/json'
            },
        });
        res.json(data);
    } catch (error) {
        console.error('Error calling the text-to-image API:', error.message);
        res.status(500).json({ message: 'Internal server error' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
