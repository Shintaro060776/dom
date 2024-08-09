import React from 'react';
import './24.css';

const BlogArticle24 = () => {

    const pythonCode = `
name: CI/CD Pipeline

on:
  push:
    branches:
      - main

env:
  AWS_REGION: ap-northeast-1

jobs:
  terraform:
    runs-on: ubuntu-latest
    env:
      AWS_ACCESS_KEY_ID: AWS_ACCESS_KEY_ID
      AWS_SECRET_ACCESS_KEY: AWS_SECRET_ACCESS_KEY

    steps:
      - name: Check Out Repository
        uses: actions/checkout@v2

      - name: 'Setup Terraform'
        uses: hashicorp/setup-terraform@v1
        with:
          terraform_version: 1.0.11

      - name: 'Terraform Init'
        run: terraform init
        working-directory: terraform_configuration

      - name: 'List files in terraform_configuration directory'
        run: |
          ls -alh terraform_configuration/
          pwd
          ls

      - name: 'Terraform Plan'
        run: terraform plan -no-color
        working-directory: terraform_configuration

      - name: 'Terraform Apply'
        run: terraform apply -auto-approve -no-color
        working-directory: terraform_configuration
        env:
          TF_VAR_openai_api_key: TF_VAR_OPENAI_API_KEY

  deploy:
    needs: terraform
    runs-on: ubuntu-latest

    steps:
      - name: checkout code
        uses: actions/checkout@v2

      - name: setup nodejs
        uses: actions/setup-node@v2
        with:
            node-version: '14'

      - name: install dependencies for blog app
        run: |
          cd blog/myapp
          npm install

      - name: build the blog react app
        run: |
          cd blog/myapp
          npm run build

      - name: build the fluid app
        run: |
          cd fluid
          npm install
          npm run build

      - name: emotion_backend
        run: |
          cd emotion_backend
          npm install

      - name: emotion_frontend
        run: |
          cd emotion/frontend
          npm install
          npm run build

      - name: realtime_backend
        run: |
          cd realtime_backend
          npm install
  
      - name: realtime
        run: |
          cd realtime/myapp
          npm install
          npm run build

      - name: well_backend
        run: |
          cd well_backend
          npm install
  
      - name: well-generate
        run: |
          cd well-generate/myapp
          npm install
          npm run build

      - name: claim_backend
        run: |
          cd claim_backend
          npm install
  
      - name: claim-generate
        run: |
          cd claim_generation/myapp
          npm install
          npm run build

      - name: dalle_backend
        run: |
          cd dalle_backend
          npm install
  
      - name: dalle
        run: |
          cd dalle/myapp
          npm install
          npm run build

      - name: gpt4_backend
        run: |
          cd gpt4_backend
          npm install
  
      - name: gpt4
        run: |
          cd gpt4/myapp
          npm install
          npm run build

      - name: speech_backend
        run: |
          cd speech_backend
          npm install
  
      - name: speech
        run: |
          cd speech1/myapp
          npm install
          npm run build

      - name: image2video_backend
        run: |
          cd image2video_backend
          npm install
  
      - name: image2video
        run: |
          cd image2video/myapp
          npm install
          npm run build

      - name: image2image_backend
        run: |
          cd image2image_backend
          npm install
  
      - name: image2image
        run: |
          cd image2image1/myapp
          npm install
          npm run build

      - name: text2speech_backend
        run: |
          cd text2speech_backend
          npm install
  
      - name: text2speech
        run: |
          cd text2speech/myapp
          npm install
          npm run build

      - name: text2image_backend
        run: |
          cd text2image_backend
          npm install
  
      - name: text2image
        run: |
          cd text2image/myapp
          npm install
          npm run build

      - name: ailab_backend
        run: |
          cd ailab_backend
          npm install
  
      - name: ailab
        run: |
          cd ailab1/myapp
          npm install
          npm run build

      - name: music_backend
        run: |
          cd music_backend
          npm install
  
      - name: music1
        run: |
          cd music1/myapp
          npm install
          npm run build

      - name: events_backend
        run: |
          cd events_backend
          npm install
  
      - name: events
        run: |
          cd events/myapp
          npm install
          npm run build

      - name: searchandreplace1_backend
        run: |
          cd searchandreplace1_backend
          npm install
  
      - name: searchandreplace1
        run: |
          cd searchandreplace1/myapp
          npm install
          npm run build

      - name: smokefree_backend
        run: |
          cd smokefree_backend
          npm install
  
      - name: smokefree
        run: |
          cd smokefree/myapp
          npm install
          npm run build

      - name: capture_backend
        run: |
          cd capture_backend
          npm install
  
      - name: capture1
        run: |
          cd capture1/myapp
          npm install
          npm run build

      - name: prompt_gen_backend
        run: |
          cd prompt_gen_backend
          npm install
  
      - name: prompt_gen
        run: |
          cd prompt_gen/myapp
          npm install
          npm run build

      - name: map_backend
        run: |
          cd map_backend
          npm install
  
      - name: map
        run: |
          cd map/myapp
          npm install
          npm run build

      - name: deploy to ec2
        env:
          PRIVATE_KEY: SSH_PRIVATE_KEY
          SERVER_IP: SERVER_IP 
        run: |
          echo "$PRIVATE_KEY" > deploy_key
          chmod 600 deploy_key

          ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no ec2-user@$SERVER_IP '
            sudo yum update -y
            sudo amazon-linux-extras install nginx1 -y
            sudo systemctl start nginx
            sudo systemctl enable nginx

            sudo mkdir -p /usr/share/nginx/html/blog
            sudo mkdir -p /usr/share/nginx/html/fluid
            sudo chown -R ec2-user:ec2-user /usr/share/nginx/html
          '

          ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no ec2-user@$SERVER_IP 'sudo systemctl restart nginx'

          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/blog/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/blog/

          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/fluid/dist/ ec2-user@$SERVER_IP:/usr/share/nginx/html/fluid/

          # Deploy emotion app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/emotion_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/emotion_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/emotion/frontend/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/emotion/

          # Deploy realtime app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/realtime_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/realtime_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/realtime/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/realtime/

          # Deploy well app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/well_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/well_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/well-generate/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/well-generate/

          # Deploy claim app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/claim_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/claim_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/claim_generation/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/claim_generation/

          # Deploy dalle app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/dalle_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/dalle_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/dalle/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/dalle/

          # Deploy gpt4 app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/gpt4_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/gpt4_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/gpt4/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/gpt4/

          # Deploy speech app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/speech_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/speech_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/speech1/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/speech1/

          # Deploy image2video app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/image2video_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/image2video_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/image2video/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/image2video/

          # Deploy image2image app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/image2image_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/image2image_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/image2image1/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/image2image1/

          # Deploy text2speech app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/text2speech_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/text2speech_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/text2speech/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/text2speech/

          # Deploy text2image app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/text2image_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/text2image_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/text2image/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/text2image/

          # Deploy ailab app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/ailab_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/ailab_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/ailab1/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/ailab1/

          # Deploy music app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/music_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/music_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/music1/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/music1/

          # Deploy events app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/events_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/events_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/events/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/events/

          # Deploy searchandreplace1 app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/searchandreplace1_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/searchandreplace1_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/searchandreplace1/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/searchandreplace1/

          # Deploy smokefree app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/smokefree_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/smokefree_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/smokefree/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/smokefree/

          # Deploy capture app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/capture_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/capture_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/capture1/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/capture1/

          # Deploy prompt gen app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/prompt_gen_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/prompt_gen_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/prompt_gen/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/prompt_gen/

          # Deploy map app
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/map_backend/ ec2-user@$SERVER_IP:/usr/share/nginx/html/map_backend/
          rsync -avz -e "ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no" /home/runner/work/dom/dom/map/myapp/build/ ec2-user@$SERVER_IP:/usr/share/nginx/html/map/

          ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no ec2-user@$SERVER_IP 'sudo systemctl restart nginx'

          ssh -i deploy_key -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no ec2-user@$SERVER_IP '

          #   cd /usr/share/nginx/html/realtime_backend
          #   pm2 restart 1
              cd /usr/share/nginx/html/text2image_backend
              pm2 restart 2
              cd /usr/share/nginx/html/ailab_backend
              pm2 restart 3
              cd /usr/share/nginx/html/music_backend
              pm2 restart 4
              cd /usr/share/nginx/html/events_backend
              pm2 restart 5
              cd /usr/share/nginx/html/searchandreplace1_backend
              pm2 restart 6
              cd /usr/share/nginx/html/smokefree_backend
              pm2 restart 7
              cd /usr/share/nginx/html/capture_backend
              pm2 restart 8
              /usr/share/nginx/html/prompt_gen_backend
              pm2 restart 9
              /usr/share/nginx/html/map_backend
              pm2 restart 10
          '

          rm -f deploy_key
        `;

    return (
        <div className='App'>
            <img src='/blog/CICD.png' alt='CICD' className='header-image' />
            <div className='page-title'>
                <h1>Explanation of CICD</h1>
            </div>
            <div className='page-date'>
                <p>2024/8/9</p>
            </div>
            <div className='paragraph'>
                <p>
                    Explanation of CICD<br /><br />

                    今回は、アプリケーションのDeployを自動化している、CICDのコードについて、以下に記載します。<br /><br />

                    <br /><br /><span className="highlight">Pipeline Name and Trigger</span><br /><br />
                    name: パイプラインの名前です。<br /><br />
                    on: このパイプラインがトリガーされる条件です。ここではmainブランチへのpushイベントでトリガーされます。<br /><br />

                    <br /><br /><span className="highlight">Environment Variables</span><br /><br />
                    env: 環境変数を定義しています。AWS_REGIONはAWSリソースが配置されるリージョン(ここではap-northeast-1)を指定しています。<br /><br />

                    <br /><br /><span className="highlight">Terraform Job</span><br /><br />
                    runs-on: このジョブが実行される環境(ubuntu-latest)を指定します。<br /><br />
                    env: AWSの認証情報をsecretsから取得しています。<br /><br />

                    <br /><br /><span className="highlight">Steps</span><br /><br />
                    Check Out Repository: リポジトリのコードをチェックアウトします。<br /><br />
                    Setup Terraform: Terraformのバージョン1.0.11をセットアップします。<br /><br />
                    Terraform Init: terraform initコマンドを使用して、Terraformの初期設定を行います。<br /><br />
                    List files in terraform_configuration directory: 指定されたディレクトリのファイルをリストします。<br /><br />
                    Terraform Plan: terraform planコマンドでインフラの変更計画を確認します。<br /><br />
                    Terraform Apply: terraform applyコマンドで計画された変更を適用します。TF_VAR_openai_api_keyはTerraformに渡される環境変数です。<br /><br />

                    <br /><br /><span className="highlight">Deploy Job</span><br /><br />
                    needs: terraformジョブが完了した後にこのジョブが実行されることを示します。<br /><br />
                    runs-on: このジョブが実行される環境(ubuntu-latest)を指定します。                    <br /><br />

                    <br /><br /><span className="highlight">Steps</span><br /><br />
                    checkout code: リポジトリのコードを再度チェックアウトします。<br /><br />
                    setup nodejs: Node.jsのバージョン14をセットアップします。<br /><br />
                    install dependencies: 各アプリケーションの依存関係をnpm installでインストールします。<br /><br />
                    build the app: Reactアプリケーションなどをnpm run buildでビルドします。<br /><br />

                    <br /><br /><span className="highlight">Deployment to EC2</span><br /><br />
                    env: EC2インスタンスへの接続に使用するSSHキーとサーバーIPアドレスを設定しています。
                    run: sshを使用してEC2インスタンスに接続し、必要なパッケージをインストールし、Nginxサーバーをセットアップします。<br /><br />
                    rsyncを使用してローカルのビルド成果物をEC2インスタンスにデプロイします。<br /><br />
                    最後に、pm2 restartを使ってバックエンドサービスを再起動します。<br /><br />

                    <br /><br /><span className="highlight">Overall Flow</span><br /><br />
                    このCI/CDパイプラインは、Terraformを使ってインフラを構築し、次にNode.jsベースの複数のフロントエンドアプリケーションとバックエンドサービスをビルドしてEC2インスタンスにデプロイします。<br /><br />
                    すべてのビルドとデプロイの手順が自動化されており、mainブランチにコードがプッシュされるたびに実行されます。<br /><br />

                    <br /><br />以下は、忘備録として、CICD(yaml)のコードの説明を記載します。<br /><br />

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

export default BlogArticle24;
