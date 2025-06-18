from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from model import FaceDetection

app = Flask(__name__)
model = FaceDetection()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    img_data = data.get('image')

    if not name or not img_data:
        return jsonify({'success': False, 'msg': 'Invalid data'})

    # Decode image
    img_bytes = base64.b64decode(img_data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    msg, color = model.register_face(name, img)
    return jsonify({'success': True, 'msg': msg, 'color': color})

if __name__ == '__main__':
    app.run(debug=True)
