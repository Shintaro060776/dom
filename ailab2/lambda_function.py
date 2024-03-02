import json
import boto3
import requests
import os
from botocore.exceptions import ClientError
import time

def lambda_handler(event, context):
    s3_client = boto3.client('s3')
    bucket_name = 'ailab20090317'

    try:
        result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='images/', MaxKeys=1, StartAfter='images/')
        latest_key = sorted(result.get('Contents', []), key=lambda x: x['LastModified'], reverse=True)[0]['Key']

        file_obj = s3_client.get_object(Bucket=bucket_name, Key=latest_key)
        file_content = file_obj['Body'].read()

        api_url = "https://www.ailabapi.com/api/image/effects/ai-anime-generator"
        headers = {'ailabapi-api-key': os.environ['AILAB_API_KEY']}
        response = requests.post(api_url, headers=headers, files={'image': file_content})
        response_data = response.json()

        task_id = response_data.get('task_id')
        query_url = "https://www.ailabapi.com/api/common/query-async-task-result?task_id=" + task_id

        while True:
            query_response = requests.get(query_url, headers=headers)
            query_data = query_response.json()
            task_status = query_data.get('task_status')

            if task_status == 2:
                resutl_url = query_data['data']['result_url']
                break
            elif task_status != 1:
                raise Exception("AILAB API task failed or timed out")
            time.sleep(1)

        transformed_image = requests.get(result_url).content
        transformed_key = f"transformed/{latest_key.split('/')[-1]}"
        s3_client.put_object(Bucket=bucket_name, Key=transformed_key, Body=transformed_image)

    except ClientError as e:
        print(f"Error accessing S3: {e}")
        return {'statusCode': 500, 'body': json.dumps("Error accessing S3")}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'statusCode': 500, 'body': json.dumps("Unexpected error occurred")}

    return {
        'statusCode': 200,
        'body': json.dumps({'message': "Image processed successfully", 'transformedKey': transformed_key})
    }