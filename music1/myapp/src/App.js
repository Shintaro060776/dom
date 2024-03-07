import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'

function App() {
  const [musicInfo, setMusicInfo] = useState({ title: '', rating: '' });
  const [selectedFile, setSelectedFile] = useState(null);
  const [musicList, setMusicList] = useState([]);

  useEffect(() => {
    fetchMusicList();
  }, []);

  const fetchMusicList = async () => {
    const response = await axios.get('http://52.68.145.180/api/music');
    setMusicList(response.data);
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const musicData = { title: musicInfo.title, rating: musicInfo.rating, fileName: selectedFile.name };
      const presignedUrlResponse = await axios.post('http://52.68.145.180/api/music-presigned', musicData);
      const { url } = presignedUrlResponse.data;

      console.log('Presigned URL:', url);

      await axios.put(url, selectedFile, {
        headers: {
          'Content-Type': selectedFile.type,
          'x-amz-acl': 'bucket-owner-full-control',
        },
      });

      await fetchMusicList();

      setMusicInfo({ title: '', rating: '' });
      setSelectedFile(null);
      alert('音楽情報がアップロードされました');
    } catch (error) {
      console.error('アップロード中にエラーが発生しました:', error);
      alert('音楽情報のアップロードに失敗しました');
    }
  };

  return (
    <div>
      <header>
        <h1>Music Rating</h1>
        <a href='http://52.68.145.180/'>Go to Top</a>
      </header>
      <main>
        <form onSubmit={handleSubmit}>
          <input
            type='text'
            value={musicInfo.title}
            onChange={(e) => setMusicInfo({ ...musicInfo, title: e.target.value })}
            placeholder='タイトル'
            required
          />
          <input
            type='text'
            value={musicInfo.rating}
            onChange={(e) => setMusicInfo({ ...musicInfo, rating: e.target.value })}
            placeholder='評価'
            required
          />
          <input
            type='file'
            onChange={handleFileChange}
            required
          />
          <button type='submit'>Upload</button>
        </form>
        <section>
          {musicList.map((music, index) => (
            <div key={index}>
              <h3>{music.title}</h3>
              <p>評価: {music.rating}</p>
              {music.imageUrl && <img src={music.imageUrl} alt={music.title} />}
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}

export default App;
