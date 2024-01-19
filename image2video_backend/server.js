const express = require('express');
const axios = require('axios');
const multer = require('multer');
const app = express();

const upload = multer();

app.post('/api/image2video', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            throw new Error("No file uploaded with field name 'image'");
        }

        const formData = new FormData();
        formData.append('image', req.file.buffer, req.file.originalname);

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