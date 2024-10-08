import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [userId, setUserId] = useState('');            // ユーザーID
  const [taskName, setTaskName] = useState('');        // タスク名
  const [actionType, setActionType] = useState('');    // タスクの進捗状況（action_type）
  const [responseData, setResponseData] = useState(''); // サーバーからのレスポンス
  const [displayedText, setDisplayedText] = useState(''); // タイピング風に表示されるテキスト
  const [error, setError] = useState(null);            // エラーメッセージ

  const apiGatewayUrl = 'https://9spa8q89d8.execute-api.ap-northeast-1.amazonaws.com/prod/timemanagement';  // API GatewayのURL

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
      const jsonData = {
        user_id: userId,
        task_name: taskName,
        action_type: actionType
      };

      const response = await axios.post(apiGatewayUrl, jsonData, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      setResponseData(JSON.stringify(response.data, null, 2)); // レスポンスデータをJSON文字列に変換
      setDisplayedText(''); // タイピング表示用のテキストをリセット
      setError(null);
    } catch (err) {
      console.error(err);
      setError('エラーが発生しました: ' + err.message);
      setResponseData('');
      setDisplayedText(''); // エラーメッセージ時もテキストをクリア
    }
  };

  return (
    <div className='App'>
      <h1>時間管理とパフォーマンス分析アプリ</h1>

      {/* ユーザーID入力フォーム */}
      <input
        type="text"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        placeholder="ユーザーIDを入力してください"
      />

      {/* タスク入力フォーム */}
      <input
        type="text"
        value={taskName}
        onChange={(e) => setTaskName(e.target.value)}
        placeholder="タスク名を入力してください"
      />

      {/* タスク進捗状況の選択 */}
      <select value={actionType} onChange={(e) => setActionType(e.target.value)}>
        <option value="">進捗状況を選択</option>
        <option value="start">開始</option>
        <option value="complete">完了</option>
      </select>

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