import './App.css';

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
    <div className="App">
      <header className="header">
        <h1 className="title"><a href="next-lb-477304975.ap-northeast-1.elb.amazonaws.com">My Personal Blog</a></h1>
        <nav className="navigation">
          <a href="/blog">Blog</a>
          <a href="/about">About</a>
        </nav>
      </header>
      <main className="blog-container">
        <h2 className="blog-title">Blog</h2>
        <section className="blog-grid">
          {blogEntries.map(entry => (
            <article key={entry.id} className="blog-entry">
              <a href={entry.link} target="_blank" rel="noopener nofererer">
                <img src={entry.thumbnail} alt={entry.title} />
              </a>
              <p className="blog-date">{entry.date}</p>
              <h3 className="blog-theme">{entry.title}</h3>
            </article>
          ))}
        </section>
      </main>
    </div>
  );
}

export default App;
