const express = require('express');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const app = express();

const upload = multer();

app.post('/api/image2video', express.json(), async (req, res) => {
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