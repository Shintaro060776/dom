import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [taskName, setTaskName] = useState('');        // タスク名
  const [status, setStatus] = useState('');            // タスクの進捗状況
  const [file, setFile] = useState(null);              // アップロードするファイル
  const [responseData, setResponseData] = useState(''); // サーバーからのレスポンス
  const [displayedText, setDisplayedText] = useState(''); // タイピング風に表示されるテキスト
  const [error, setError] = useState(null);            // エラーメッセージ

  const apiGatewayUrl = 'xxxxxxxxxxxxxxxxxxxxxxxxxx';  // API GatewayのURL

  useEffect(() => {
    let typingIndex = 0;
    let intervalId;

    // レスポンスがある場合に、タイピング表示を開始
    if (responseData) {
      intervalId = setInterval(() => {
        setDisplayedText((prev) => prev + responseData[typingIndex]);
        typingIndex += 1;

        // 全ての文字が表示されたら停止
        if (typingIndex === responseData.length) {
          clearInterval(intervalId);
        }
      }, 100); // 100ms毎に1文字表示
    }

    // クリーンアップ関数
    return () => {
      clearInterval(intervalId);
    };
  }, [responseData]);

  // タスクの進行状況をサーバーに送信する
  const submitTask = async () => {
    try {
      const formData = new FormData();
      formData.append('task_name', taskName);
      formData.append('status', status);
      if (file) {
        formData.append('file', file);
      }

      const response = await axios.post(apiGatewayUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResponseData(JSON.stringify(response.data, null, 2)); // レスポンスデータをJSON文字列に変換
      setDisplayedText(''); // タイピング表示用のテキストをリセット
      setError(null);
    } catch (err) {
      setError('エラーが発生しました: ' + err.message);
      setResponseData('');
      setDisplayedText(''); // エラーメッセージ時もテキストをクリア
    }
  };

  return (
    <div className='App'>
      <h1>時間管理とパフォーマンス分析アプリ</h1>

      {/* タスク入力フォーム */}
      <input
        type="text"
        value={taskName}
        onChange={(e) => setTaskName(e.target.value)}
        placeholder="タスク名を入力してください"
      />

      {/* タスク進捗状況の選択 */}
      <select value={status} onChange={(e) => setStatus(e.target.value)}>
        <option value="">進捗状況を選択</option>
        <option value="start">開始</option>
        <option value="in_progress">進行中</option>
        <option value="complete">完了</option>
      </select>

      {/* ファイルアップロード */}
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />

      {/* サーバーに送信 */}
      <button onClick={submitTask}>タスクを送信</button>

      {/* エラーがある場合 */}
      {error && <p>{error}</p>}

      {/* サーバーからのレスポンス表示 */}
      <div>
        <h2>サーバーからのレスポンス:</h2>
        {/* タイピング風に表示されるテキスト */}
        <pre>{displayedText}</pre>
      </div>
    </div>
  );
}

export default App;