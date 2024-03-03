import json
import boto3
import requests
import os
from botocore.exceptions import ClientError
import time

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    bucket_name = 'ailab20090317'

    try:
        result = s3.list_objects_v2(Bucket=bucket_name)
        contents = result.get('Contents', [])

        if not contents:
            print("No objects found in the bucket")
            return {
                'statusCode': 404,
                'body': json.dumps('No objects found in the bucket')
            }
            
        latest_key = sorted(contents, key=lambda x: x['LastModified'], reverse=True)[0]['Key']
        print(f"Latest file key: {latest_key}")

        file_obj = s3.get_object(Bucket=bucket_name, Key=latest_key)
        file_content = file_obj['Body'].read()

        api_url = "https://www.ailabapi.com/api/image/effects/ai-anime-generator"
        headers = {'ailabapi-api-key': os.environ['AILAB_API_KEY']}

        payload = {'type': 'boy-5'}

        files = {
            'image': (latest_key, file_content, 'image/jpeg'),
            'task_type': ('', 'async'),
            'index': ('', '0')
        }

        response = requests.post(api_url, headers=headers, data=payload, files=files)

        try:
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                response_data = response.json()
                print(f"AILAB API Response: {response_data}")
            else:
                print(f"Unexpected Content-Type: {content_type}")
                response_data = None
        except json.JSONDecodeError:
            print("JSON解析エラーが発生しました")
            response_data = None

        if response_data:
            task_id = response_data.get('task_id')
            if task_id is None:
                print("task_id is None, full response:", response_data)
                return {'statusCode': 500, 'body': json.dumps('Unexpected error: task_id is None')}
        
            query_url = "https://www.ailabapi.com/api/common/query-async-task-result?task_id=" + task_id

            while True:
                query_response = requests.get(query_url, headers=headers)
                query_data = query_response.json()
                task_status = query_data.get('task_status')

                if task_status == 2:
                    result_url = query_data['data']['result_url']
                    break
                elif task_status != 1:
                    raise Exception("AILAB API task failed or timed out")
                time.sleep(1)

            transformed_image = requests.get(result_url).content
            transformed_key = f"transformed/{latest_key.split('/')[-1]}"
            s3.put_object(Bucket=bucket_name, Key=transformed_key, Body=transformed_image)
        else:
            print("レスポンスデータが取得出来ませんでした")
            return {'statusCode': 500, 'body': json.dumps("Unexpected error: response data is missing")}

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