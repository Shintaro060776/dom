import React from 'react';
import './About.css';

const AboutPage = () => {
    return (
        <div className="App">
            <img src="20231017_18_24_0.png" alt="firstone" className="header-image" />
            <div className="page-title">
                <h1>About</h1>
            </div>
            <div className="paragraph">
                <p>
                    Shintaro Hashimoto(橋本慎太郎)<br /><br />
                    現在は、SRE/Devopsエンジニアとして、働いています。<br /><br />
                    お仕事の依頼は、こちらから、宜しくお願い致します。<br /><br />
                    shintaro060776@gmail.com<br /><br />
                    <a href="https://twitter.com/area51439213784" target="_blank" rel="noopener noreferrer">
                        <img src="/icons8-twitter-48.png" alt="twitter" className="sns-icon" />
                    </a>
                    <a href="https://www.instagram.com/shintaro20090317_2/" target="_blank" rel="noopener noreferrer">
                        <img src="/icons8-instagram-48.png" alt="instagram" className="sns-icon" />
                    </a>
                    <a href="https://github.com/Shintaro060776/dom" target="_blank" rel="noopener noreferrer">
                        <img src="/icons8-github-48.png" alt="github" className="sns-icon" />
                    </a>
                </p>
            </div>
        </div>
    );
};

export default AboutPage;