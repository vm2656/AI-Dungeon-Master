import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [conversation, setConversation] = useState([]);
  const [gameState, setGameState] = useState(null);

  useEffect(() => {
    resetGame();
  }, []);

  const resetGame = async () => {
    try {
      const response = await fetch('http://localhost:8000/reset', {
        method: 'POST',
      });
      const data = await response.json();
      setConversation([{ role: 'ai', content: data.initial_context }]);
      setGameState({ turns: 0, context: data.initial_context });
    } catch (error) {
      console.error('Error resetting game:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setConversation(prev => [...prev, { role: 'player', content: input }]);

    try {
      const response = await fetch('http://localhost:8000/generate_response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input }),
      });

      const data = await response.json();

      setConversation(prev => [...prev, { role: 'ai', content: data.response }]);
      setGameState(data.game_state);
    } catch (error) {
      console.error('Error:', error);
    }

    setInput('');
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Dungeon Master</h1>
        <div className="game-info">
          <p>Turns: {gameState?.turns || 0}</p>
          <p>Context: {gameState?.context || 'Loading...'}</p>
        </div>
        <div className="conversation">
          {conversation.map((message, index) => (
            <div key={index} className={message.role}>
              <strong>{message.role === 'ai' ? 'DM:' : 'You:'}</strong> {message.content}
            </div>
          ))}
        </div>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter your action..."
          />
          <button type="submit">Send</button>
        </form>
        <button onClick={resetGame}>Reset Game</button>
      </header>
    </div>
  );
}

export default App;