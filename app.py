# app.py
from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io

app = Flask(__name__)

# --- Helper function to read an image from a request file ---
def read_image_from_request(file_storage):
    """Reads an image from a Flask FileStorage object and decodes it with OpenCV."""
    in_memory_file = io.BytesIO()
    file_storage.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# --- Main Web Route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- API Endpoint for All Image Processing Tools ---
@app.route('/api/process', methods=['POST'])
def api_process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    img = read_image_from_request(file)
    tool = request.form.get('tool')
    processed_img = None

    # --- All existing tool logic is preserved here ---
    if tool == 'grayscale':
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif tool == 'enhance':
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        processed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    elif tool == 'blur':
        processed_img = cv2.GaussianBlur(img, (21, 21), 0)
    elif tool == 'sketch':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(gray_img)
        blur = cv2.GaussianBlur(invert, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blur)
        processed_img = cv2.divide(gray_img, inverted_blur, scale=256.0)
    elif tool == 'face_detect':
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        processed_img = img
    elif tool == 'posterize':
        K = 8
        Z = img.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        processed_img = res.reshape((img.shape))
    # Add other tools here if needed
    else:
        processed_img = img # Default to original if tool is unknown

    if processed_img is not None:
        # Send processed image back as PNG for display consistency
        _, img_encoded = cv2.imencode('.png', processed_img)
        return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')
        
    return jsonify({'error': 'Processing failed.'}), 400

# --- NEW: API Endpoint for Downloading in Different Formats ---
@app.route('/api/download', methods=['POST'])
def download_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image data received.'}), 400

    file = request.files['image']
    img = read_image_from_request(file)
    
    file_format = request.form.get('format', 'png').lower()
    if file_format not in ['png', 'jpg', 'webp']:
        return jsonify({'error': 'Invalid format.'}), 400

    # --- Encode the image to the selected format ---
    if file_format == 'jpg':
        mimetype = 'image/jpeg'
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        success, buffer = cv2.imencode('.jpg', img, encode_param)
    elif file_format == 'webp':
        mimetype = 'image/webp'
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), 95]
        success, buffer = cv2.imencode('.webp', img, encode_param)
    else: # Default to PNG
        mimetype = 'image/png'
        success, buffer = cv2.imencode('.png', img)

    if not success:
        return jsonify({'error': 'Failed to encode image.'}), 500

    # --- Send the file to the user for download ---
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype=mimetype,
        as_attachment=True,
        download_name=f'processed_image.{file_format}'
    )

if __name__ == '__main__':
    app.run(debug=True)

