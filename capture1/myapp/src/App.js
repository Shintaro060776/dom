import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [images, setImages] = useState([]);
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const fetchImages = async () => {
      try {
        const response = await axios.get('http://52.68.145.180/api/capture2-get-images');
        setImages(response.data);
      } catch (error) {
        console.error('Error fetching images:', error);
      }
    };

    fetchImages();
  }, []);

  const startDrawing = () => setIsDrawing(true);
  const stopDrawing = () => setIsDrawing(false);

  const draw = (event) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
  };

  const saveImage = async () => {
    const canvas = canvasRef.current;
    const dataURL = canvas.toDataURL('image/png');
    const binary = atob(dataURL.split(',')[1]);
    const array = [];

    for (let i = 0; i < binary.length; i++) {
      array.push(binary.charCodeAt(i));
    }
    const blobData = new Blob([new Uint8Array(array)], { type: 'image/png' });

    try {
      const response = await axios.post('http://52.68.145.180/api/capture1-presigned-url', { fileType: 'image/png' });
      const { presignedUrl, fileName } = response.data;

      await axios.put(presignedUrl, blobData, {
        headers: {
          'Content-Type': 'image/png',
          'x-amz-acl': 'bucket-owner-full-control',
        },
      });

      console.log('Image uploaded:', fileName);
      const fetchImages = async () => {
        try {
          const response = await axios.get('http://52.68.145.180/api/capture2-get-images');
          setImages(response.data);
        } catch (error) {
          console.error('Error fetching images:', error);
        }
      };
      fetchImages();
    } catch (error) {
      if (error.response) {
        console.error('Error saving image:', error.response.data);
      } else if (error.request) {
        console.error('Error saving image: No response received', error.request);
      } else {
        console.error('Error saving image:', error.message);
      }
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Image Gallery</h1>
        <a href="http://52.68.145.180/">トップページに戻る</a>
      </header>
      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width="500"
          height="500"
          className="canvas"
          onMouseDown={startDrawing}
          onMouseUp={stopDrawing}
          onMouseMove={draw}
        ></canvas>
        <div className="button-container">
          <button onClick={saveImage}>Save</button>
          <button onClick={clearCanvas}>Clear</button>
        </div>
      </div>
      <div className="image-gallery">
        {images.map(image => (
          <div key={image.id} className="image-wrapper">
            <img src={image.url} alt={image.id} className="image" />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;