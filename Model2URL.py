from flask import Flask, send_from_directory
import os

app = Flask(__name__)

MODEL_PATH = 'path_to_model'

@app.route('/model', methods=['GET'])
def get_model():
    # Invia il modello TensorFlow
    return send_from_directory(os.path.dirname(MODEL_PATH), os.path.basename(MODEL_PATH))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
