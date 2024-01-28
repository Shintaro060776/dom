import json
from openai import OpenAI
import os
import boto3
from openai.error import OpenAIError


def lambda_handler(event, context):

    s3_client = boto3.client('s3')

    bucket_name = "text2speech20090317"
    s3_file_path = "speech.mp3"

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        user_input = event.get("user_input", "")

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": user_input}
                ]
            )
        except OpenAIError as e:
            print("Error with OpenAI GPT-4 API:", e)
            return {"statusCode": 500, "body": json.dumps({"error": "OpenAI GPT-4 API error:" + str(e)})}

        generated_text = response['choices'][0]['message']['content']

        speech_file_path = "/tmp/speech.mp3"

        try:
            speech_response = client.audio.speech_create(
                model="tts-1",
                voice="alloy",
                input=generated_text
            )
        except OpenAIError as e:
            print("Error with OpenAI Text to Speech API:", e)
            return {"statusCode": 500, "body": json.dumps({"error": "OpenAI Text to Speech API error: " + str(e)})}

        speech_response.stream_to_file(speech_file_path)

        try:
            with open(speech_file_path, 'rb') as speech_file:
                s3_client.upload_fileobj(
                    speech_file, bucket_name, s3_file_path)
        except boto3.exceptions.S3UploadFailedError as e:
            print("Error uploading file to S3:", e)
            return {"statusCode": 500, "body": json.dumps({"error": "S3 upload error:" + str(e)})}

        audio_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_path}"

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": generated_text,
                "audio_url": audio_url
            })
        }

    except Exception as e:
        print(e)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
