const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const AWS = require('aws-sdk');

const app = express();
const port = 10000;
const s3 = new AWS.S3()
const BUCKET_NAME = 'image2video20090317'
const VIDEO_FOLDER = 'video/';

app.use(bodyParser.json());

app.post('/api/image2video', async (req, res) => {
    try {
        const response = await axios.post(
            'https://ixcnxwety3.execute-api.ap-northeast-1.amazonaws.com/prod/stabilityai1',
            req.body
        );
        res.json(response.data);
    } catch (error) {
        console.error('Error on forwarding request to API Gateway:', error.message);
        res.status(500).json({ 'axios': 'Internal Server Error' });
    }
});

app.get('/api/latest_video', async (req, res) => {
    try {
        const params = {
            Bucket: BUCKET_NAME,
            Prefix: VIDEO_FOLDER,
        };

        const data = await s3.listObjectsV2(params).promise();
        const videoFiles = data.Contents;

        const latestFile = videoFiles.reduce((latest, file) => {
            return (!latest || file.LastModified > latest.LastModified) ? file : latest;
        }, null);

        if (!latestFile) {
            return res.status(404).json({ error: "No videos found" });
        }

        const videoUrl = `https://${BUCKET_NAME}.s3.amazonaws.com/${latestFile.Key}`;
        res.json({ videoUrl });
    } catch (error) {
        console.error("Error retrieving latest video:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});