// App.jsx

import React, { useState } from 'react';

const App = () => {
    const [file1, setFile1] = useState(null);
    const [file2, setFile2] = useState(null);
    const [matchResult, setMatchResult] = useState(null);

    const handleFile1Change = (e) => {
        setFile1(e.target.files[0]);
    };

    const handleFile2Change = (e) => {
        setFile2(e.target.files[0]);
    };

    const handleCompare = async () => {
        if (!file1 || !file2) return;

        const formData = new FormData();
        formData.append('file1', file1);
        formData.append('file2', file2);

        try {
            const response = await fetch('http://localhost:5000/api/facerec', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                setMatchResult(data.match);
            } else {
                console.error('Face recognition request failed');
            }
        } catch (error) {
            console.error('Error in face recognition:', error);
        }
    };

    return (
        <div>
            <input type="file" onChange={handleFile1Change} />
            <input type="file" onChange={handleFile2Change} />
            <button onClick={handleCompare}>Compare Faces</button>
            {matchResult !== null && (
                <p>Match Result: {matchResult ? 'Same person' : 'Different person'}</p>
            )}
        </div>
    );
};

export default App;
