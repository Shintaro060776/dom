import json
import boto3
from datetime import datetime
import openai
import os

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('SmokeFreeUserData')

def lambda_handler(event, context):
    print("Received event:", event)
    try:
        data = json.loads(event['body'])
        print("Parsed data:", data)
        user_id = data.get('user_id', 'default_user_id')
        smoke_free_days = data['smoke_free_days']
        cigarettes_not_smoked = data['cigarettes_not_smoked']
        money_saved = data['money_saved']

        timestamp = datetime.now().isoformat()
        item = {
            'user_id': user_id,
            'timestamp': timestamp,
            'smoke_free_days': smoke_free_days,
            'cigarettes_not_smoked': cigarettes_not_smoked,
            'money_saved': money_saved
        }

        table.put_item(Item=item)

        openai.api_key = os.environ.get("OPENAI_API_KEY")
        user_prompt = f"How can I stay motivated after being smoke-free for {smoke_free_days} days, not smoking {cigarettes_not_smoked} cigarettes, and saving {money_saved} dollars?"

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )

        ai_response = response.choices[0].message.content

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data updated successfully',
                'ai_response': ai_response
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Error processing request',
                'message': str(e)
            })
        }