import React from 'react';
import './12.css';

const BlogArticle12 = () => {

    const pythonCode = `
    import requests

    # ###########################################################
    
    # 秘匿情報と、APIのエンドポイント設定
    email = 'shashimoto@gnext.co.jp'
    global_api_key = 'XXXXXXXXXXXXX'
    zone_id = 'XXXXXXXXXXXXX'
    rule_id = 'XXXXXXXXXXXXX'
    ruleset_id = 'XXXXXXXXXXXXX'
    update_rule_api_url = f'https://api.cloudflare.com/client/v4/zones/{zone_id}/rulesets/{ruleset_id}/rules/{rule_id}'
    
    # ###########################################################
    
    headers = {
        'X-Auth-Email': email,
        'X-Auth-Key': global_api_key,
        'Content-Type': 'application/json'
    }
    
    #  既存のルールに設定されている、IPアドレスを読み込んで、スペースで、IPアドレスを区切る
    with open('existing_ips.txt', 'r') as file:
        ip_list_exist = file.read().splitlines() # file.read()で、読み込んだ文字列を、リストに変換して、行単位で、データを扱えるようにする
    
    ip_list_str = ' '.join(ip_list_exist)
    
    # 条件式を、f-stringで囲う
    existing_expression = f'(http.host contains "XXXXXXXXXXXXX(rulename)" and ip.src in {{{ip_list_str}}})'
    
    # 既存のルール情報
    existing_action = 'skip'  
    existing_action_parameters = {'ruleset': 'current'}
    existing_ips = set(existing_expression.split("ip.src in {")[1].split("}")[0].split()) # 『ip.src in {』で、既存の、Expressionを分割して、分割したリストの、2番目の要素(IPアドレスのリスト)を取得する。さらに、取得した文字列を、『}』で分割して、『{}』の中に、記載されているIPアドレスのリストが、抽出される。さらに、『split()』である、スペースで分割して、各IPアドレスを、個別の要素として、リストを作成する。最後に、setで、集合に変換して、重複しているIPアドレスを、除外する
    rule_name = 'XXXXXXXXXXXXX(description)'
    
    ip_list_file = 'ip_list.txt'
    
    # 追加するIPアドレスを読み込み、スペースで区切るようにする
    with open(ip_list_file, 'r') as file:
        ip_list = file.read().splitlines()
    
    # 追加するIPアドレスに、重複がないかチェックし、新しいIPアドレスを追加する
    new_ips = [ip for ip in ip_list if ip not in existing_ips] # 追加するIPアドレスである、ip_listの要素に対して、ループさせて、既存のIPアドレス(existing_ips)に、含まれていない場合に、new_ipsに、IPアドレスを追加する
    if new_ips:
        updated_ips = existing_ips.union(set(new_ips)) # (set(new_ips))で、セットに変換して、重複している可能性のある、追加するIPアドレスを、チェックする。そして、unionで、既存のIPアドレス(existing_ips)と、追加するIPアドレスの、集合を作成する。
        new_expression = f"(http.host contains "XXXXXXXXXXXXX(rulename)" and ip.src in {{{' '.join(updated_ips)}}})" # 上記の、updated_ipsに格納されている、IPアドレスを、スペースで区切って、Expressionに、記載していく
       
        # ルールの中身を更新
        update_data = {
            'action': existing_action,  
            'action_parameters': existing_action_parameters,
            'expression': new_expression,
            'description': rule_name,
        }
       
        # CloudflareのAPIに対して、Patchで、IPアドレスを、既存のルールに追加する
        response = requests.patch(update_rule_api_url, headers=headers, json=update_data)
    
        # APIから、返ってくるステータスによって、ターミナルに、処理結果を、出力させる
        if response.status_code == 200:
            print("Rule updated successfully")
        else:
            print(f"Failed to update rule. Status Code: {response.status_code}, Response: {response.text}")
    else:
        print("No new IPs to add.")
        `;

    return (
        <div className='App'>
            <img src='/blog/20240127_04_50_0.png' alt='eleventh' className='header-image' />
            <div className='page-title'>
                <h1>仕事で実装するコード化の一例</h1>
            </div>
            <div className='page-date'>
                <p>2023/09/15</p>
            </div>
            <div className='paragraph'>
                <p>
                    仕事で実装するコード化の一例<br /><br />

                    今回は、仕事で、実装するコード化について、紹介しようと思います。インフラ作業であっても、可能な限り、コード化させています。その一例を、紹介したいと思います。<br /><br />

                    <br /><br /><span className="highlight">Cloudflareへの、大量の、IPアドレス追加</span><br /><br />
                    リクエスターから、依頼があって、大量(数百個)のIPアドレスを追加して、送信元から、アクセス許可させてほしい、という依頼がありました。<br /><br />
                    手作業で、一つ一つ、GUIから、IPアドレスを追加するのは、得策ではないと思われます。<br /><br />
                    Cloudflare側のドキュメントにて、既存のルールに対して、APIを叩けば、設定変更可能と記載されていていたので、この提供されているAPIを利用して、IPアドレスを追加しようと思いました。<br /><br />
                    エンジニアである以上、業務を最短で終わらせられる方法を、チョイスすべきであると、個人的に思っています。<br /><br />
                    また、今回は、Pythonでコードを書きましたが、他の別の方法があるのかもしれません。<br /><br />
                    しかし、何故、今回(今までの、コード化も含めて)、Pythonでコードを書いて、Cloudflare側のAPIを叩いて、大量のIPアドレスを追加しようと思ったかと言うと、もう一つ理由があります。<br /><br />
                    それは、技術力の向上です。<br /><br />
                    手作業で作業するのは、正直、エンジニアでなくても、未経験の人であっても、作業可能です。<br /><br />
                    未経験の人でも出来ることを、エンジニアが、ファーストチョイスとして、選択するのは、良い案だとは、思いません。<br /><br />
                    そういった、個人的なポリシーに従って、今まで、可能な限り、コード化させて、業務を完了させるスピードを、向上させています。<br /><br />
                    当然、コード化のメリットは、次回、類似の依頼が舞い込んだ際にも、最短で、完了させられるメリットもあるので、コード化を選択しない手はないですね。<br /><br />

                    <br /><br />以下は、忘備録として、Pythonのコードの説明を記載します。<br /><br />

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

export default BlogArticle12;
