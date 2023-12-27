import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AboutPage from './AboutPage';
import BlogArticle from './1';
import BlogArticle2 from './2';
import BlogArticle3 from './3';
import BlogArticle4 from './4';
import BlogArticle5 from './5';

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