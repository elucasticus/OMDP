from flask import Flask, request
import requests
import time
from onnx_second_inference_flask import onnx_search_and_run_second_half
from onnx_third_inference_flask import onnx_search_and_run_third_half
import numpy as np
import json

def run(onnx_path, EP_list, device):
    app = Flask(__name__)

    @app.route("/")
    def root():
        return "<h1>Hello There</h1>"

    @app.route("/status")
    def return_server_status():
        return "<h1>Operational</h1>"

    @app.route("/endpoint", methods=["POST", "GET"])
    def endpoint():
        data = json.load(request.files['data'])
        data["result"] = np.load(request.files['document']).tolist()
        arrival_time = time.time()
        returnData = onnx_search_and_run_third_half(onnx_path, None, data, None, EP_list, device)
        returnData["Outcome"] = "Success"
        returnData["arrival_time"] = arrival_time
        returnData["result"] = returnData["result"].tolist()
        return returnData

    @app.route("/checkpoint", methods=["POST", "GET"])
    def checkpoint():
        data = json.load(request.files['data'])
        data["result"] = np.load(request.files['document']).tolist()
        data = onnx_search_and_run_second_half(onnx_path, None, data, None, EP_list, device)
        np.save("input_check", data["result"])
        del data["result"]
        files = [
            ('document', ("input_check.npy", open("input_check.npy", 'rb'), 'application/octet')),
            ('data', ('data', json.dumps(data), 'application/json')),
        ]
        #Send the Intermediate Tensors to the server
        print("Sending the intermediate tensors to the server...")
        server_url = "http://127.0.0.1:3000/endpoint"
        response = requests.post(server_url, files=files).json()
        return response
    
    return app

