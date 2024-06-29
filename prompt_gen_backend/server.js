const express = require('express');
const axios = require('axios');
const AWS = require('aws-sdk');

const app = express();
const s3 = new AWS.S3();
const port = 25000;
const bucketName = 'prompt-gen-20090317';
const prefix = 'generated_images/';

const API_GATEWAY_URL = 'https://XXXXXXXXXXXXXXXXXXXXXXXXX';

app.use(express.json());

app.post('/api/prompt-gen', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_URL, req.body);
        res.send(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).send('Failed to generate image');
    }
});

app.get('/api/get-prompt-gen', async (req, res) => {
    try {
        const params = {
            Bucket: bucketName,
            Prefix: prefix
        };

        const data = await s3.listObjectsV2(params).promise();
        const latestImage = data.Contents.sort((a, b) => b.LastModified - a.LastModified)[0];
        const imageParams = {
            Bucket: bucketName,
            Key: latestImage.Key
        };

        const imageData = await s3.getObject(imageParams).promise();
        res.writeHead(200, { 'Content-Type': 'image/webp' });
        res.write(imageData.Body, 'binary');
        res.end(null, 'binary');
    } catch (error) {
        console.error(error);
        res.status(500).send('Failed to get image');
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});

