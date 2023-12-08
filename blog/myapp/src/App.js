import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AboutPage from './AboutPage';
import BlogArticle from './1';

const blogEntries = [
  {
    id: 1,
    title: "Nice to meet you",
    date: "2023/01/03",
    thumbnail: "20231006_18_13_0.png",
    path: "/blog1/1"
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