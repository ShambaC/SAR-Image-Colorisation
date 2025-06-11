from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from utils.get_info import get_country, get_season, calc_hemisphere, get_region

import subprocess
import sys
import platform
import os
import utils

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

        # Define paths
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        tool_dir = os.path.join(base_dir, "Dataset Tools/copernicus_api")
        ldm_dir = os.path.join(base_dir, "LDM")

        country = get_country(lat, long)
        hemisphere = calc_hemisphere(lat)
        season = get_season(country, f"{img_date}T12:00:00Z", hemisphere)
        temp_region = get_region(lat)

        if platform.system() == "Windows" :
            activate_script = os.path.join(ldm_dir, "Scripts", "activate.bat")
            command = f'cmd.exe /c "{activate_script} && python infer_model.py --config config/sar_config.yaml --mode single --sar_image ../Application/backend/{image_path} --region {temp_region} --season {season} --output colorized_{filename}"'
        else :
            activate_script = os.path.join(ldm_dir, "bin", "activate")
            command = f'/bin/bash /c "source {activate_script} && python infer_model.py --config config/sar_config.yaml --mode single --sar_image ../Application/backend/{image_path} --region {temp_region} --season {season} --output colorized_{filename}"'

        result = subprocess.run(
            command,
            shell=True,
            cwd=ldm_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return jsonify({
            "status": "success",
            "message": "Image generated successfully!",
            "imageUrl": f"colorized_{filename}"
        }), 200
    
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": e.stderr.strip()}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/image/<string:filename>')
def serve_image(filename):
    return send_file(f'../../LDM/{filename}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
