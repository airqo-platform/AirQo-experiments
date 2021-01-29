from flask import Flask
import calibration
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'