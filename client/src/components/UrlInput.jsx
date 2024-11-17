import React, { useState } from 'react';


const UrlInput = () => {
  const [urls, setUrls] = useState(["", ""]);
  const [loading, setLoading] = useState(false);

  const handleChange = (index, value) => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  const handleProcess = async () => {
    setLoading(true);
    const response = await fetch('/process-urls', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ urls }), 
    });

    const result = await response.json();
    alert(result.message || 'Error processing URLs');
    setLoading(false);
  };

  return (
    <div className="row-container">
      <h2>Enter News Article URLs</h2>
      {urls.map((url, index) => (
        <input
          key={index}
          type="text"
          value={url}
          onChange={(e) => handleChange(index, e.target.value)}
          placeholder={`URL ${index + 1}`}
        />
      ))}
      <button onClick={handleProcess} disabled={loading}>
        {loading ? 'Processing...' : 'Process URLs'}
      </button>
    </div>
  );
};

export default UrlInput;
