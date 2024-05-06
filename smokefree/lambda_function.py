import json
import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('SmokeFreeUserData')

def lambda_handler(event, context):
    try:
        data = json.loads(event['body'])
        user_id = data['user_id']
        smoke_free_days = data['smoke_free_days']
        cigarettes_not_smoked = data['cigarettes_not_smoked']
        money_saved = data['money_saved']

        timestamp = datetime.now().isoformat()

        response = table.put_item(
            Item={
                'user_id': user_id,
                'timestamp': timestamp,
                'smoke_free_days': smoke_free_days,
                'cigarettes_not_smoked': cigarettes_not_smoked,
                'money_saved': money_saved
            }
        )

        return {
            'statusCode': 200,
            'body': json.dumps('Data updated successfully')
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Error processing request: ' + str(e))
        }