from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import os
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return "Backend is running."

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Get data from request
        image = request.files.get('image')
        lat = request.form.get('lat')
        long = request.form.get('long')
        img_date = request.form.get('imgDate')
        
        print(lat, long, img_date, image)

        if not image or not lat or not long or not img_date:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # Save image
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)

        # Run your ML code
        result = subprocess.run(
            ['python', 'dummy.py', image_path, lat, long, img_date],
            check=True,
            capture_output=True,
            text=True
        )

        # output_filename = result.stdout.strip()

        return jsonify({
            "status": "success",
            "message": "Image generated successfully!",
            "imageUrl": "/jk.jpg"
        })
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": e.stderr.strip()}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/jk.jpg')
def serve_jk():
    return send_file('jk.jpg')

if __name__ == '__main__':
    app.run(debug=True)
