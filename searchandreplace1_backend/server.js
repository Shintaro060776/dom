const express = require('express');
const axios = require('axios');
const AWS = require('aws-sdk');
const s3 = new AWS.S3();
const app = express();
const port = 20000;

app.use(express.json());

const API_GATEWAY_ENDPOINT = 'https://jmr1cgg0ff.execute-api.ap-northeast-1.amazonaws.com/prod/searchandreplace1';
const S3_BUCKET = 'searchandreplace20090317';
const S3_PREFIX = 'gen/';

app.post('/api/searchandreplace', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_ENDPOINT, req.body);
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).send('Internal Server Error');
    }
});

app.get('/api/latestsearchandreplace', async (req, res) => {
    try {
        const params = {
            Bucket: S3_BUCKET,
            Prefix: S3_PREFIX
        };

        const s3Response = await s3.listObjectsV2(params).promise();
        const files = s3Response.Contents;

        const latestFile = files.sort((a, b) => new Date(b.LatModified) - new Date(a.LastModified))[0];

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
