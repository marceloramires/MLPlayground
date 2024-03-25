from flask import Flask
from utils.ModelRunner import *

app = Flask(__name__)
runner = ModelRunner()

@app.route("/<name>")
def predict(name):
    return runner.predict(name) 