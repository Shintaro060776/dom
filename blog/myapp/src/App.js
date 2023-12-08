import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AboutPage from './AboutPage';

const blogEntries = [
  {
    id: 1,
    title: "Nice to meet you",
    date: "2023/01/03",
    thumbnail: "20231006_18_13_0.png",
    link: "/blog1/1.html"
  },
  {
    id: 2,
    title: "ブログのタイトル2",
    date: "2023/01/02",
    thumbnail: "20231006_18_13_0.png",
    link: "/blog2/2.html"
  },
  {
    id: 3,
    title: "ブログのタイトル3",
    date: "2023/01/02",
    thumbnail: "20231006_18_13_0.png",
    link: "/blog3/3.html"
  },
];

function App() {
  return (
    <Router>
      <div className="App">
        <header className="header">
          <h1 className="title"><Link to="/blog">My Personal Blog</Link></h1>
          <nav className="navigation">
            <Link to="/blog">Blog</Link>
            <Link to="/about">About</Link>
          </nav>
        </header>
        <Routes>
          <Route path="/about" element={<AboutPage />} />
          <Route path="/blog" element={
            <main className="blog-container">
              <h2 className="blog-title">Blog</h2>
              <section className="blog-grid">
                {blogEntries.map(entry => (
                  <article key={entry.id} className="blog-entry">
                    <Link to={entry.link} target="_blank">
                      <img src={entry.thumbnail} alt={entry.title} />
                    </Link>
                    <p className="blog-date">{entry.date}</p>
                    <h3 className="blog-theme">{entry.title}</h3>
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
