from flask import Flask, request
import requests
import time
from onnx_second_inference_flask import onnx_extract_and_run_second_half
from onnx_third_inference_flask import onnx_search_and_run_third_half
import numpy as np
import json
import onnx
import pickle
import csv

def run(onnx_file, EP_list, device):
    app = Flask(__name__)

    #Load the onnx model and extract the final output names
    onnx_model = onnx.load(onnx_file)
    end_names = []
    for i in range(len(onnx_model.graph.output)):
        end_names.append(onnx_model.graph.output[i].name)

    #Load the list of the possible split points
    with open("temp/split_layers", "rb") as fp:   # Unpickling
        split_layers = pickle.load(fp)
 

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

        #Compute arrival time 
        arrival_time = time.time()

        #Extract the input layer and its index into the list of possible splits
        input_layer_index = split_layers.index(data["splitLayer"])
        input_names = [data["splitLayer"]]

        onnx_model_file = "temp/second_half.onnx"

        with open("checkpoint_results.csv", "w", newline="") as csvfile:
            fields = ["splitPoint1", "splitPoint2", "execTime1", "execTime2", "networkingTime"]
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            row = {"splitPoint1": data["splitLayer"],
                   "splitPoint2": "",
                   "execTime1":0.,
                   "execTime2":0.,
                   "networkingTime": 0.}
            for i in range(input_layer_index + 1, len(split_layers)):
                #Find the second split point 
                name = split_layers[i]
                print("##### Output layer: %s #####" %name)
                output_names = [name]
                try:
                    ##Split the model to obtain the second submodel and compute the time needed to run it
                    up_data = onnx_extract_and_run_second_half(onnx_file, input_names, output_names, 
                                                            onnx_model_file, data, None, EP_list, device)
                    np.save("input_check", up_data["result"])
                    del up_data["result"]
                    up_data["splitLayer"] = name
                    up_data["execTime1"] = up_data["execTime2"]
                    files = [
                        ('document', ("input_check.npy", open("input_check.npy", 'rb'), 'application/octet')),
                        ('data', ('up_data', json.dumps(up_data), 'application/json')),
                    ]

                    #Send the Intermediate Tensors to the server
                    print("Sending the intermediate tensors to the server...")
                    server_url = "http://127.0.0.1:3000/endpoint"
                    departure_time = time.time()
                    response = requests.post(server_url, files=files).json() 
                    
                    #Save the results
                    row["splitPoint2"] = split_layers[i]
                    row["execTime1"] = response["execTime1"]
                    row["execTime2"] = response["execTime2"] 
                    row["networkingTime"] = response["arrival_time"] - departure_time
                    writer.writerow(row)              
                except:
                    print("Cannot extract the submodel!")
            
            #Trivial case: we don't rely on the server
            print("##### Trivial case #####")
            output_names = end_names
            returnData = onnx_extract_and_run_second_half(onnx_file, input_names, output_names, 
                                                            onnx_model_file, data, None, EP_list, device)

            #Save the results 
            row["splitPoint2"] = "end"
            row["execTime1"] = returnData["execTime2"]
            row["execTime2"] = 0.
            row["networkingTime"] = 0.
            writer.writerow(row)

            #Return the results
            returnData["Outcome"] = "Success"
            returnData["arrival_time"] = arrival_time
            returnData["result"] = returnData["result"].tolist()

        return returnData
    
    return app

