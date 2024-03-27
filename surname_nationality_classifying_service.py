from flask import Flask
from utils.ModelRunner import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
runner = ModelRunner()
cors = CORS(app)

@app.route("/<name>")
@cross_origin()
def predict(name):
    return runner.predict(name) 