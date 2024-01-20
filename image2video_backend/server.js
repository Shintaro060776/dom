const express = require('express');
const axios = require('axios');
const FormData = require('form-data');
const app = express();
const bodyParser = require('body-parser');

app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

app.post('/api/image2video', async (req, res) => {
    try {
        if (!req.body.image || !req.body.filename) {
            throw new Error("No image data or filename provided");
        }

        const imageBuffer = Buffer.from(req.body.image, 'base64');

        const formData = new FormData();
        formData.append('image', imageBuffer, { filename: req.body.filename });

        const response = await axios.post(
            'https://kgqmlycmzc.execute-api.ap-northeast-1.amazonaws.com/prd/image',
            formData,
            { headers: { ...formData.getHeaders() } }
        );
        res.json(response.data);
    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ message: error.message });
    }
});

const PORT = 10000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});