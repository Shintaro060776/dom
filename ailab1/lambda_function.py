import json
import boto3
import traceback
from botocore.client import Config

def lambda_handler(event, context):
    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
    bucket_name = 'ailab20090317'
    expiration = 300

    try:
        # eventのbodyがNoneでないかを確認し、Noneの場合は空のJSONオブジェクトとして扱う
        body_str = event.get('body', '{}')
        if body_str is None:
            body_str = '{}'
        
        # bodyの内容をログに出力
        print("Received body:", body_str)
        
        body = json.loads(body_str)

        if 'data' in body and 'fileName' in body['data']:
            file_name = body['data']['fileName']
            response = s3.generate_presigned_url('put_object',
                                                Params={
                                                    'Bucket': bucket_name,
                                                    'Key': file_name,
                                                    'ACL': 'bucket-owner-full-control'},
                                                ExpiresIn=expiration,
                                                HttpMethod="PUT")

            return {
                'statusCode': 200,
                'body': json.dumps({'url': response})
            }
        else:
            # 'fileName'が見つからない場合のエラーメッセージを改善
            print("Missing 'fileName' in the request body")
            return {
                'statusCode': 400,
                'body': json.dumps('Missing or invalid query parameters: "fileName" is required')
            }
    except json.JSONDecodeError as e:
        # JSONのパースに失敗した場合のエラーハンドリング
        print(f"JSON Decode Error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid JSON format')
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps(f'Internal Server Error: {str(e)}')
        }