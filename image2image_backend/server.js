const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const AWS = require('aws-sdk');
const s3 = new AWS.S3();
const app = express();
const port = 11000;

app.use(bodyParser.json());

const API_GATEWAY_ENDPOINT = 'https://p7b37ekuh3.execute-api.ap-northeast-1.amazonaws.com/prod/image2image1';
const S3_BUCKET = 'image2image20090317';
const S3_PREFIX = 'gen/';

app.post('/api/image2image', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_ENDPOINT, req.body);
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).send('Internal Server Error');
    }
});

app.get('/api/latest_image', async (req, res) => {
    try {
        const params = {
            Bucket: S3_BUCKET,
            Prefix: S3_PREFIX
        };

        const s3Response = await s3.listObjectsV2(params).promise();
        const files = s3Response.Contents;

        const latestFile = files.sort((a, b) => b.LastModified - a.LastModified)[0];

        if (!latestFile) {
            return res.status(404).send('No images found');
        }

        const imageUrl = `https://${S3_BUCKET}.s3.amazonaws.com/${latestFile.Key}`;
        res.json({ url: imageUrl });
    } catch (error) {
        console.error(error);
        res.status(500).send('Internal Server Error');
    }
});

app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});