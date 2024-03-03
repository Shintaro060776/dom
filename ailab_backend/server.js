const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const AWS = require('aws-sdk');

const app = express();
const port = 15000;
AWS.config.update({ region: 'ap-northeast-1' });
const s3 = new AWS.S3();
const BUCKET_NAME = 'ailab20090317';
const IMAGE_FOLDER = 'images/';

app.use(bodyParser.json());

app.post('/api/presigned-url', async (req, res) => {
    try {
        const response = await axios.post('https://86bcsyqls6.execute-api.ap-northeast-1.amazonaws.com/prod/ailab', {
            headers: {
                'Content-Type': 'application/json'
            },
            data: JSON.stringify(req.body) // リクエストボディをそのままLambdaに転送
        });
        res.json(response.data);
    } catch (error) {
        console.error('Error on forwarding request to API Gateway:', error.response ? JSON.stringify(error.response.data) : error.message);
        res.status(500).json({ error: 'Internal Server Error', details: error.response ? error.response.data : null });
    }
});

app.get('/api/latest-animated-image', async (req, res) => {
    try {
        const params = {
            Bucket: BUCKET_NAME,
            Prefix: IMAGE_FOLDER,
        };

        const data = await s3.listObjectsV2(params).promise();
        const imageFiles = data.Contents;

        const latestFile = imageFiles.reduce((latest, file) => {
            return (!latest || file.LastModified > latest.LastModified) ? file : latest;
        }, null);

        if (!latestFile) {
            return res.status(404).json({ error: "No animated images found" });
        }

        const imageUrl = `https://${BUCKET_NAME}.s3.amazonaws.com/${latestFile.Key}`;
        res.json({ imageUrl });
    } catch (error) {
        console.error("Error retrieving latest animated image:", error);
        res.status(500).json({ error: "Internal Server Error", details: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});