import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AboutPage from './AboutPage';
import BlogArticle from './1';
import BlogArticle2 from './2';
import BlogArticle3 from './3';

const blogEntries = [
  {
    id: 1,
    title: "Nice to meet you",
    date: "2023/01/03",
    thumbnail: "20231006_18_13_0.png",
    path: "/blog1/1"
  },
  {
    id: 2,
    title: "機械学習を用いた実験的Webアプリケーションについて",
    date: "2023/01/09",
    thumbnail: "20231016_12_05_0.png",
    path: "/blog2/2"
  },
  {
    id: 3,
    title: "考え中",
    date: "2023/01/14",
    thumbnail: "20231004_18_26_0.png",
    path: "/blog3/3"
  },
];

function handleTopLinkClick(event) {
  event.preventDefault();
  window.location.href = '/';
}

function handleLinkClick(event) {
  event.preventDefault();
  window.location.href = '/blog';
}

function handleAboutClick(event) {
  event.preventDefault();
  window.location.href = '/about';
}

function App() {
  return (
    <Router>
      <div className="App">
        <header className="header">
          <h1 className="title"><Link to="/blog">My Personal Blog</Link></h1>
          <nav className="navigation">
            <a href="/blog" onClick={handleLinkClick}>Blog</a>
            <a href='/about' onClick={handleAboutClick}>About</a>
            <a href='/' onClick={handleTopLinkClick}>Top</a>
          </nav>
        </header>
        <Routes>
          <Route path="/about" element={<AboutPage />} />
          <Route path="/blog1/1" element={<BlogArticle />} />
          <Route path="/blog2/2" element={<BlogArticle2 />} />
          <Route path="/blog3/3" element={<BlogArticle3 />} />
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