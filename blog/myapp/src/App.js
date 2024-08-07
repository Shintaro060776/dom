import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AboutPage from './AboutPage';
import BlogArticle from './1';
import BlogArticle2 from './2';
import BlogArticle3 from './3';
import BlogArticle4 from './4';
import BlogArticle5 from './5';
import BlogArticle6 from './6';
import BlogArticle7 from './7';
import BlogArticle8 from './8';
import BlogArticle9 from './9';
import BlogArticle10 from './10';
import BlogArticle11 from './11';
import BlogArticle12 from './12';
import BlogArticle13 from './13';
import BlogArticle14 from './14';
import BlogArticle15 from './15';
import BlogArticle16 from './16';
import BlogArticle17 from './17';
import BlogArticle18 from './18';
import BlogArticle19 from './19';
import BlogArticle20 from './20';
import BlogArticle21 from './21';
import BlogArticle22 from './22';
import BlogArticle23 from './23';

const blogEntries = [
  {
    id: 1,
    title: "Nice to meet you",
    date: "2023/01/03",
    thumbnail: "/blog/20231006_18_13_0.png",
    path: "/blog1/1"
  },
  {
    id: 2,
    title: "機械学習を用いた実験的Webアプリケーションについて",
    date: "2023/01/09",
    thumbnail: "/blog/20231016_12_05_0.png",
    path: "/blog2/2"
  },
  {
    id: 3,
    title: "歌詞生成Webアプリケーション(Sagemaker)",
    date: "2023/01/14",
    thumbnail: "/blog/20231004_18_26_0.png",
    path: "/blog3/3"
  },
  {
    id: 4,
    title: "感情分析アプリケーション",
    date: "2023/02/19",
    thumbnail: "/blog/20231018_07_37_0.png",
    path: "/blog4/4"
  },
  {
    id: 5,
    title: "Realtime Chatbot Emotional Analysis",
    date: "2023/03/27",
    thumbnail: "/blog/20231126_12_19_0.png",
    path: "/blog5/5"
  },
  {
    id: 6,
    title: "Category Classification/Text Generation",
    date: "2023/04/29",
    thumbnail: "/blog/20240105_14_01_0.png",
    path: "/blog6/6"
  },
  {
    id: 7,
    title: "Emotional Analysis/Text Generation for dealing with inquiry/claim",
    date: "2023/05/28",
    thumbnail: "/blog/20240113_12_25_0.png",
    path: "/blog7/7"
  },
  {
    id: 8,
    title: "Dalle",
    date: "2023/06/25",
    thumbnail: "/blog/20240114_12_00_0.png",
    path: "/blog8/8"
  },
  {
    id: 9,
    title: "GPT4",
    date: "2023/07/29",
    thumbnail: "/blog/20240115_10_28_0.png",
    path: "/blog9/9"
  },
  {
    id: 10,
    title: "Speech-to-Text/GPT4",
    date: "2023/08/30",
    thumbnail: "/blog/20240118_07_45_0.png",
    path: "/blog10/10"
  },
  {
    id: 12,
    title: "仕事で実装するコードの一例",
    date: "2023/09/15",
    thumbnail: "/blog/20240127_04_50_0.png",
    path: "/blog12/12"
  },
  {
    id: 13,
    title: "Image-to-Image StabilityAI",
    date: "2023/10/29",
    thumbnail: "/blog/generated_20240127_06_47_0.png",
    path: "/blog13/13"
  },
  {
    id: 14,
    title: "Text-to-Speech OpenAI",
    date: "2023/11/28",
    thumbnail: "/blog/20240128_10_30_0.png",
    path: "/blog14/14"
  },
  {
    id: 11,
    title: "Image-to-Video By StabilityAi",
    date: "2023/12/30",
    thumbnail: "/blog/20240119_12_38_0_convert.png",
    path: "/blog11/11"
  },
  {
    id: 15,
    title: "text-to-image By StabilityAi",
    date: "2024/1/28",
    thumbnail: "/blog/v1_txt2img_2024-03-02T08_23_40.823434_0.png",
    path: "/blog15/15"
  },
  {
    id: 16,
    title: "Converting to animated image By AILAB",
    date: "2024/2/29",
    thumbnail: "/blog/IMG_5355.jpeg",
    path: "/blog16/16"
  },
  {
    id: 17,
    title: "Music Rating",
    date: "2024/3/8",
    thumbnail: "/blog/music.jpg",
    path: "/blog17/17"
  },
  {
    id: 18,
    title: "Scheduling Event",
    date: "2024/4/8",
    thumbnail: "/blog/event.jpg",
    path: "/blog18/18"
  },
  {
    id: 19,
    title: "Search And Replace",
    date: "2024/4/17",
    thumbnail: "/blog/searchandreplace.png",
    path: "/blog19/19"
  },
  {
    id: 20,
    title: "SmokeFree",
    date: "2024/5/7",
    thumbnail: "/blog/smokefree.png",
    path: "/blog20/20"
  },
  {
    id: 21,
    title: "Image Gallery",
    date: "2024/6/16",
    thumbnail: "/blog/drawing.png",
    path: "/blog21/21"
  },
  {
    id: 22,
    title: "Image Generation By StabilityAI Latest Model",
    date: "2024/7/1",
    thumbnail: "/blog/StabilityAILatest.png",
    path: "/blog22/22"
  },
  {
    id: 23,
    title: "Marathon Tracker",
    date: "2024/8/7",
    thumbnail: "/blog/marathon.png",
    path: "/blog23/23"
  },
];

function handleTopLinkClick(event) {
  event.preventDefault();
  window.location.href = '/';
}

function App() {
  return (
    <Router>
      <div className="App">
        <header className="header">
          <h1 className="title"><Link to="/blog">My Personal Blog</Link></h1>
          <nav className="navigation">
            <Link to="/blog">Blog</Link>
            <Link to="/about">About</Link>
            <a href='/' onClick={handleTopLinkClick}>Top</a>
          </nav>
        </header>
        <Routes>
          <Route path="/about" element={<AboutPage />} />
          <Route path="/blog1/1" element={<BlogArticle />} />
          <Route path="/blog2/2" element={<BlogArticle2 />} />
          <Route path="/blog3/3" element={<BlogArticle3 />} />
          <Route path="/blog4/4" element={<BlogArticle4 />} />
          <Route path="/blog5/5" element={<BlogArticle5 />} />
          <Route path="/blog5/5" element={<BlogArticle5 />} />
          <Route path="/blog6/6" element={<BlogArticle6 />} />
          <Route path="/blog7/7" element={<BlogArticle7 />} />
          <Route path="/blog8/8" element={<BlogArticle8 />} />
          <Route path="/blog9/9" element={<BlogArticle9 />} />
          <Route path="/blog10/10" element={<BlogArticle10 />} />
          <Route path="/blog12/12" element={<BlogArticle12 />} />
          <Route path="/blog13/13" element={<BlogArticle13 />} />
          <Route path="/blog14/14" element={<BlogArticle14 />} />
          <Route path="/blog11/11" element={<BlogArticle11 />} />
          <Route path="/blog15/15" element={<BlogArticle15 />} />
          <Route path="/blog16/16" element={<BlogArticle16 />} />
          <Route path="/blog17/17" element={<BlogArticle17 />} />
          <Route path="/blog18/18" element={<BlogArticle18 />} />
          <Route path="/blog19/19" element={<BlogArticle19 />} />
          <Route path="/blog20/20" element={<BlogArticle20 />} />
          <Route path="/blog21/21" element={<BlogArticle21 />} />
          <Route path="/blog22/22" element={<BlogArticle22 />} />
          <Route path="/blog23/23" element={<BlogArticle23 />} />
          <Route path="/blog" element={
            <main className="blog-container">
              <h2 className="blog-title">Blog</h2>
              <section className="blog-grid">
                {blogEntries.map(entry => (
                  <article key={entry.id} className="blog-entry">
                    <Link to={entry.path}>
                      <img src={entry.thumbnail} alt={entry.title} />
                      <div>
                        <h3>{entry.title}</h3>
                        <p>{entry.date}</p>
                      </div>
                    </Link>
                  </article>
                ))}
              </section>
            </main>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App;