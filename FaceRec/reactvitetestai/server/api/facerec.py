from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

@app.route('/api/facerec', methods=['POST'])
def face_recognition_endpoint():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Missing files in request'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    # Perform face recognition using face_recognition library
    # Example code (modify as per face_recognition library usage):
    image1 = face_recognition.load_image_file(file1)
    image2 = face_recognition.load_image_file(file2)
    print(image1)
    print(image2)
    
    encoding1 = face_recognition.face_encodings(image1)[0]
    print(encoding1)
    encoding2 = face_recognition.face_encodings(image2)[0]
    print(encoding2)

    results = face_recognition.compare_faces([encoding1], encoding2)
    print(results)
    print(results[0])
    
    # Convert boolean result to integer (1 for True, 0 for False)
    match_result = 1 if results[0] else 0

    return jsonify({'match': match_result})

if __name__ == '__main__':
    app.run(debug=True)
