const express = require('express');
const axios = require('axios');
const AWS = require('aws-sdk');
const cors = require('cors');

AWS.config.update({
    region: 'ap-northeast-1'
});

const app = express();
const dynamodb = new AWS.DynamoDB.DocumentClient();
const port = 27000;

const API_GATEWAY_URL = 'https://c9ojd7eo7a.execute-api.ap-northeast-1.amazonaws.com/prod/map';

app.use(cors());
app.use(express.json());

app.post('/api/save-route', async (req, res) => {
    try {
        console.log('Request Body:', req.body);
        const response = await axios.post(API_GATEWAY_URL, req.body);
        console.log('API Gateway Response:', response.data);
        res.send(response.data);
    } catch (error) {
        console.error('Error saving route:', error);
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
        console.error('Error getting routes:', error);
        res.status(500).send('Failed to get routes');
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});