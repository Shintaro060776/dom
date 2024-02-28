import React from 'react';
import './11.css';

const BlogArticle11 = () => {

    const pythonCode = `
    import json
    import boto3
    import requests
    import os
    import logging
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    
    def lambda_handler(event, context):
        logger.info(f"Received event: {event}")
    
        if 'Records' in event:
            s3 = boto3.client('s3')
            bucket_name = event['Records'][0]['s3']['bucket']['name']
            object_key = event['Records'][0]['s3']['object']['key']
    
            try:
                response = s3.get_object(Bucket=bucket_name, Key=object_key)
                image_data = response['Body'].read()
    
                api_url = 'https://api.stability.ai/v2alpha/generation/image-to-video'
                api_key = os.environ['STABILITY_API_KEY']
    
                headers = {'Authorization': f'Bearer {api_key}'}
                data = {'seed': 0, 'cfg_scale': 2.5, 'motion_bucket_id': 40}
                file_extension = object_key.split('.')[-1]
                mime_type = 'image/jpeg' if file_extension.lower() == 'jpg' else 'image/png'
                files = {'image': (object_key, image_data, mime_type)}
    
                api_response = requests.post(
                    api_url, headers=headers, data=data, files=files)
                logger.info(
                    f"API Response Status Code: {api_response.status_code}")
                logger.info(f"API Response Headers: {api_response.headers}")
                logger.info(f"API Response Body: {api_response.text}")
    
                api_response.raise_for_status()
    
                generation_id = api_response.json().get('id')
                if generation_id:
                    sfn_client = boto3.client('stepfunctions')
                    state_machine_arn = 'arn:aws:states:ap-northeast-1:715573459931:stateMachine:VideoGenerationStateMachine'
                    sfn_response = sfn_client.start_execution(
                        stateMachineArn=state_machine_arn,
                        input=json.dumps({'generation_id': generation_id})
                    )
                    return {
                        'statusCode': 200,
                        'body': json.dumps({'generation_id': generation_id, 'executionArn': sfn_response['executionArn']})
                    }
                else:
                    return {
                        'statusCode': 500,
                        'body': json.dumps({'error': 'Failed to start video generation'})
                    }
    
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                return {
                    'statusCode': 500,
                    'body': json.dumps({'error': str(e)})
                }
    
        elif 'generation_id' in event:
            return {
                'statusCode': 200,
                'body': json.dumps(event)
            }
    
        else:
            return {'statusCode': 200, 'body': json.dumps({'message': 'Triggered by Step Functions, no action required'})}
    
        `;

    return (
        <div className='App'>
            <img src='/blog/20240119_12_38_0_convert.png' alt='eleventh' className='header-image' />
            <div className='page-title'>
                <h1>Image-to-Video By StabilityAi</h1>
            </div>
            <div className='page-date'>
                <p>2023/12/30</p>
            </div>
            <div className='paragraph'>
                <p>
                    Image-to-Video By StabilityAi<br /><br />

                    StabilityAIのAPIを利用して、画像データを、動画に変換して、フロント側で、再生させています<br /><br />

                    <img src='/blog/system11.png' alt='eleventhsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロントエンド</span><br /><br />
                    ユーザーは、Reactであるフロントにて、画像ファイルを、選択します。<br /><br />
                    画像ファイルは、Reactアプリケーションによって、状態管理されます。<br /><br />
                    「Generate Video」ボタンをクリックすると、Reactアプリケーションは、バックエンドのAPIに、リクエストを送信し、プリサインドURLを取得します。<br /><br />
                    取得したプリサインドURLを使用して、フロントエンドから、直接、S3バケットにファイルを、アップロードします。<br /><br />

                    <br /><br /><span className="highlight">バックエンド</span><br /><br />
                    フロントエンドからのリクエストを受け、Lambda関数が、AWS SDKを使用して、S3バケットに対して、プリサインドURLを生成します。<br /><br />
                    このURLは、特定のファイルを、アップロードするために、一時的に、使用されます。<br /><br />
                    S3バケットに、ファイルがアップロードされると、これがトリガー(S3イベント)となり、StabilityAIのAPIをPOSTするための、Lambda関数が実行されます。<br /><br />
                    Lambda関数が、生成された動画を、StabilityaiのAPIに対して、GETして、取得します<br /><br />
                    生成された動画を、S3に保存します。<br /><br />

                    <br /><br /><span className="highlight">外部サービス(Stabilityai)</span><br /><br />
                    Lambda関数は、取得した画像ファイルを、Stabilityaiの、Image-to-Video変換モデルに、送信します。<br /><br />
                    Image-to-Video変換モデルモデルは、画像を動画に変換し、その結果を、Lambda関数に返します。<br /><br />

                    <br /><br /><span className="highlight">結果の保存と取得</span><br /><br />
                    Lambda関数は、生成された動画を、S3に保存します。<br /><br />

                    <br /><br /><span className="highlight">フロントエンドの動画表示</span><br /><br />
                    ユーザーが、「Get Video」ボタンをクリックすると、Reactアプリケーションは、Nodejsを経由して、S3バケットに保存された、動画ファイルを、取得しにいきます<br /><br />
                    取得した動画は、フロントにて、表示されます。<br /><br />

                    <br /><br /><span className="highlight">通信の要約</span><br /><br />
                    React(画像Upload) == Nodejs == API Gateway == Lambda(プリサインURL発行) == React == S3 == Lambda(StabilityaiへのPost) == StabilityAI == StepFunction(StabilityaiのHTTPステータスを定期的なインターバルで確認) == Lambda(StabilityAI APIへのGET)<br /><br />

                    <br /><br /><span className="highlight">Stepfunctionのロジック</span><br /><br />
                    StabilityaiのAPIに、Postする、Lambda関数のステータスを確認して、202のスタータスを返すようであれば、再度、ステータス確認のロジックに、処理が戻ります。そして、200のステータスを返す場合は、StabilityaiのAPIに、GETを実行する関数に、処理が遷移する様に、実装しました<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/stabilityai.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video><br /><br />

                    <br /><br />以下は、忘備録として、Lambdaのコードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                            <pre><code>{pythonCode}</code></pre>
                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle11;
