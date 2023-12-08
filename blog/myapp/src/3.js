import React from 'react';
import './3.css';

const BlogArticle3 = () => {
    return (
        <div className='App'>
            <img src='/blog/20231004_18_26_0.png' alt='third' className='header-image' />
            <div className='page-title'>
                <h1>考え中</h1>
            </div>
            <div className='page-date'>
                <p>2023/01/14</p>
            </div>
            <div className='paragraph'>
                <p>
                    実験サイトで作った、機械学習を用いたサービスの概要を、以下に記載します。<br /><br />

                    <span className="highlight">①動くヌコ</span><br /><br />
                    URL : http://neilaeden.com/predict/<br /><br />
                    アプリの概要：ユーザーがフォーム上に、ネコの絵を書いて、その書いた絵を、アニメーション(動かす)させるというWebアプリケーション<br /><br />
                    難易度：★★★★☆<br /><br />
                    雑感 : Web上に、なかなか手書きのネコの画像が無いから、完成度は、そこまで高くないかな。推論処理は動くし、エラーも出ていないから、ま、いいか、という感じ。
                    何故、ネコだけにしたかというと、ネコ以外の学習データ(犬、鳥、魚など)も取り入れると、機械学習(AI)サーバーの学習時間が増大して、費用が高くなるから、限定的にしたという経緯です。スマホで、フォームに絵を書けるけど、書きづらいから、PCのブラウザから、どうぞ
                </p>
            </div>
        </div>
    );
};

export default BlogArticle3;
