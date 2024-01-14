const express = require('express');
const axios = require('axios');
const app = express();
const port = 7000;

app.use(express.json());

const cors = require('cors');
app.use(cors());

app.get('/', (req, res) => {
    res.send('Nodejs server is running');
});

app.post('/api/dalle', async (req, res) => {
    try {
        const prompt = req.body.text;
        const response = await axios.post('https://vcfxizwoc4.execute-api.ap-northeast-1.amazonaws.com/prod/openai', { prompt: prompt });
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).send('Error while calling OpenAI API');
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
