import json
import boto3
from datetime import datetime
import openai
import os

# DynamoDBに接続
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('UserTasks')  # DynamoDBのテーブル名を設定

# 環境変数からOpenAI APIキーを取得
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI APIキーが設定されていません。Lambdaの環境変数を確認してください。")

# OpenAI APIキーを設定
openai.api_key = openai_api_key

def lambda_handler(event, context):
    try:
        # イベントデータから操作タイプ（開始または完了）を取得
        action_type = event.get('action_type', 'start')  # デフォルトは'start'
        
        if action_type == 'start':
            return start_task(event)
        elif action_type == 'complete':
            return complete_task(event)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps('無効な action_type が指定されています')
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'エラーが発生しました: {str(e)}')
        }

# タスク開始処理
def start_task(event):
    try:
        user_id = event.get('user_id')
        task_name = event.get('task_name')
        start_time = event.get('start_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        if not user_id or not task_name:
            return {
                'statusCode': 400,
                'body': 'user_id または task_name がありません'
            }

        # DynamoDBにタスクを記録
        table.put_item(
            Item={
                'user_id': user_id,
                'task_name': task_name,
                'start_time': start_time,
                'status': 'in_progress'
            }
        )

        return {
            'statusCode': 200,
            'body': f'Task "{task_name}" を開始しました'
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'タスク開始時にエラーが発生しました: {str(e)}'
        }

# タスク完了処理
def complete_task(event):
    try:
        user_id = event.get('user_id')
        task_name = event.get('task_name')
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not user_id or not task_name:
            return {
                'statusCode': 400,
                'body': 'user_id または task_name がありません'
            }

        # DynamoDBから該当のタスクを取得
        response = table.get_item(
            Key={
                'user_id': user_id,
                'task_name': task_name
            }
        )
        
        if 'Item' not in response:
            return {
                'statusCode': 404,
                'body': 'タスクが見つかりません'
            }

        task = response['Item']
        start_time_str = task['start_time']
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        end_time_obj = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

        # 所要時間を計算
        time_spent = (end_time_obj - start_time).total_seconds() / 60  # 分単位で計算

        # DynamoDBに完了情報を更新
        table.update_item(
            Key={
                'user_id': user_id,
                'task_name': task_name
            },
            UpdateExpression='SET end_time = :end_time, #status = :status, time_spent = :time_spent',
            ExpressionAttributeNames={
                '#status': 'status'  # 'status'は予約語なのでExpressionAttributeNamesを使用
            },
            ExpressionAttributeValues={
                ':end_time': end_time,
                ':status': 'completed',
                ':time_spent': time_spent
            }
        )

        # OpenAI APIを使ってフィードバックを生成
        feedback = generate_feedback(task_name, time_spent)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Task "{task_name}" を完了しました',
                'time_spent': time_spent,
                'feedback': feedback
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'タスク完了時にエラーが発生しました: {str(e)}'
        }

# OpenAI APIでフィードバックを生成（`gpt-3.5-turbo` モデルを使用）
def generate_feedback(task_name, time_spent):
    try:
        messages = [
            {"role": "system", "content": "あなたはタスク管理のコーチであり、ユーザーが効率的にタスクをこなせるように助言します。"},
            {"role": "user", "content": f"ユーザーがタスク '{task_name}' を {time_spent:.2f} 分で完了しました。次のタスクを効率的に完了するためのアドバイスをください。"}
        ]
        
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # OpenAI APIのレスポンスをチェック
        if 'choices' in completion and len(completion['choices']) > 0:
            feedback = completion['choices'][0]['message']['content']
        else:
            feedback = "OpenAI APIから適切なフィードバックを取得できませんでした。"

    except Exception as e:
        feedback = f"フィードバックの生成に失敗しました: {str(e)}"
    
    return feedback