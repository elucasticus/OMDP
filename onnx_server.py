from flask import Flask, request
import requests
import time
from onnx_second_inference_flask import onnx_search_and_run_second_half
from onnx_third_inference_flask import onnx_search_and_run_third_half
import numpy as np
import json
import onnx

def run(onnx_file, EP_list, device):
    app = Flask(__name__)

    #Load the onnx model and extract the final output names
    onnx_model = onnx.load(onnx_file)
    end_names = []
    for i in range(len(onnx_model.graph.output)):
        end_names.append(onnx_model.graph.output[i].name)

    @app.route("/")
    def root():
        return "<h1>Hello There</h1>"

    @app.route("/status")
    def return_server_status():
        return "<h1>Operational</h1>"

    @app.route("/endpoint", methods=["POST", "GET"])
    def endpoint():
        #Receive the incoming data
        data = json.load(request.files['data'])
        data["result"] = np.load(request.files['document']).tolist()

        #Compute arrival time 
        arrival_time = time.time()

        #Split the model to obtain the third submodel
        onnx_model_file = "temp/third_half.onnx"
        input_names = [data["splitLayer"]]
        output_names = end_names
        onnx.utils.extract_model(onnx_file, onnx_model_file, input_names, output_names)

        #Compute the time needed to run the third submodel
        returnData = onnx_search_and_run_third_half(None, onnx_model_file, data, None, EP_list, device)

        #Return the results
        returnData["Outcome"] = "Success"
        returnData["arrival_time"] = arrival_time
        returnData["result"] = returnData["result"].tolist()
        return returnData

    @app.route("/checkpoint", methods=["POST", "GET"])
    def checkpoint():
        #Receive the incoming data
        data = json.load(request.files['data'])
        data["result"] = np.load(request.files['document']).tolist()

        #TEST
        name = "sequential/mobilenetv2_1.00_160/block_15_add/add:0"

        #Split the model to obtain the second submodel
        onnx_model_file = "temp/second_half.onnx"
        input_names = [data["splitLayer"]]
        output_names = [name]
        onnx.utils.extract_model(onnx_file, onnx_model_file, input_names, output_names)

        #Compute the time needed to run the second submodel
        data = onnx_search_and_run_second_half(None, onnx_model_file, data, None, EP_list, device)
        np.save("input_check", data["result"])
        del data["result"]
        data["splitLayer"] = name
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

