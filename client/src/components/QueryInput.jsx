import React, { useState } from 'react';

const QueryInput = () => {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    setLoading(true);
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt: query }), 
    });

    const result = await response.json();
    setAnswer(result.response || 'No response from bot');
    setLoading(false);
  };

  return (
    <div className='row-container' style={{ flexWrap: 'wrap' }}>
      <h2>Ask a Question</h2>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter your question"
      />
      <button onClick={handleQuery} disabled={loading}>
        {loading ? 'Fetching Answer...' : 'Submit'}
      </button>
      
      {answer && (
        <div className='output-container'>
          <h3>Answer:</h3>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
};

export default QueryInput;
