import json
from openai import OpenAI
import os
import boto3
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):

    s3_client = boto3.client('s3')

    bucket_name = "text2speech20090317"
    s3_file_path = "speech.mp3"

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        user_input = event.get("user_input", "")

        try:
            GPT_MODEL = "gpt-4"
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": user_input},
            ]
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0
            )

            generated_text = response.choices[0].message.content
            logger.info(f"AI Response: {generated_text}")

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise e

        speech_file_path = "/tmp/speech.mp3"

        try:
            speech_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=generated_text
            )
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise e

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
