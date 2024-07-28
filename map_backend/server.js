const express = require('express');
const axios = require('axios');
const AWS = require('aws-sdk');

const app = express();
const dynamodb = new AWS.DynamoDB.DocumentClient();
const port = 27000;

const API_GATEWAY_URL = 'xxxxxxxxxxxxxxxxx';

app.use(express.json());

app.post('/api/save-route', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_URL, req.body);
        res.send(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).send('Failed to save route');
    }
});

app.get('/api/get-routes', async (req, res) => {
    const userId = req.query.userId;

    if (!userId) {
        return res.status(400).send('Missing userId query parameter');
    }

    const params = {
        TableName: 'Routes',
        KeyConditionExpression: 'userId = :userId',
        ExpressionAttributeValues: {
            ':userId': userId
        }
    };

    try {
        const data = await dynamodb.query(params).promise();
        res.send(data.Items);
    } catch (error) {
        console.error(error);
        res.status(500).send('Failed to get routes');
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});