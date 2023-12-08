import React from 'react';
import './1.css';

const BlogArticle = () => {
    return (
        <div className='App'>
            <img src='/blog/20231006_18_13_0.png' alt='firstone' className='header-image' />
            <div className='page-title'>
                <h1>Nice to meet you</h1>
            </div>
            <div className='page-date'>
                <p>2023/01/03</p>
            </div>
            <div className='paragraph'>
                <p>
                    大学を卒業後、約10年間SIer及び社内SEとしてネットワークエンジニアの職務を果たし、NW環境の構築や運用を経験してきました。また、Globalチームとの打ち合わせを通じて、英語でプロジェクトを進める能力を身につけました。<br /><br /><span
                        className="highlight">DevOpsエンジニアとして活動しており、専門領域としては、IaCツールであるTerraformやCloudFormationを使用したインフラの設計と実装、AWSを中心としたクラウドサービスの適用、構成管理ツールであるAnsibleやChefを用いた自動化等が挙げられます。</span><br /><br />また、シェルスクリプトやPythonを使用したカスタムスクリプトの作成にも熟練しており、Dockerを活用したコンテナ環境の構築と管理の経験も豊富です。<br /><br />
                    GitHubActionsを使用したCI/CDパイプラインの実装経験もあります。<br /><br />
                    さらに、フロントエンド開発においてはReactを用いたユーザーインターフェースの構築、バックエンド開発ではNode.jsを活用したサーバーサイドのロジックの実装に関する実務経験があります。また、HTML、CSS、JavaScriptを駆使したウェブページのデザインと構築のスキルも身につけています。<br /><br />
                    これら多岐にわたるスキルと経験を活かし、チームと協力して効果的かつ効率的なソリューションを提供し続けることを常に目指しています。<br /><span
                        className="highlight">現在注目している技術は、AIとブロックチェーンで、これらを活用したサービスの実装を目指しています。</span><br /><br />
                    <br /><br />
                    このWebサイトは以下の技術で実装されています。
                    <br /><br />
                    バックエンド PHP、Nodejs、Python<br /><br />
                    フロントエンド HTML, CSS, Javascript, React<br /><br />
                    データベース DynamoDB<br /><br />
                    環境／クラウド AWS(VPC, EC2, SES, IAM, API Gateway, Lambda、Cloudwatch、S3、VPC<br /><br />
                    Endpoint、Route53、LoadBalancer、CloudFront、Braket、Sagemaker、ECR、ECS、EKS)<br /><br />
                    ツール／その他 CI/CD(Github Actions)、Terraform、Docker、Shell Script、Cloudformation<br /><br />
                </p>
            </div>
        </div>
    );
};

export default BlogArticle;
