from flask import Flask, request
import requests
import time
from onnx_second_inference_flask import onnx_extract_and_run_second_half
from onnx_third_inference_flask import onnx_search_and_run_third_half
import numpy as np
import json
import onnx
import logging
import csv
import os

def isNodeAnInitializer(onnx_model, node):
  '''
  Check if the node passed as argument is an initializer in the network.

  :param onnx_model: the already imported ONNX Model
  :param node: node's name
  :returns: True if the node is an initializer, False otherwise
  '''
  # Check if input is not an initializer, if so ignore it
  for i in range(len(onnx_model.graph.initializer)):
    if node == onnx_model.graph.initializer[i].name:
      return True

  return False

def onnx_get_true_inputs(onnx_model):
  '''
  Get the list of TRUE inputs of the ONNX model passed as argument. 
  The reason for this is that sometimes "onnx.load" interprets some of the static initializers 
  (such as weights and biases) as inputs, therefore showing a large list of inputs and misleading for instance
  the fuctions used for splitting.

  :param onnx_model: the already imported ONNX Model
  :returns: a list of the true inputs
  '''
  input_names = []

  # Iterate all inputs and check if they are valid
  for i in range(len(onnx_model.graph.input)):
    nodeName = onnx_model.graph.input[i].name
    # Check if input is not an initializer, if so ignore it
    if isNodeAnInitializer(onnx_model, nodeName):
      continue
    else:
      input_names.append(nodeName)
  
  return input_names

def run(onnx_file, EP_list, device):
    app = Flask(__name__)

    #now we will Create and configure logger 
    logging.basicConfig(filename="std.log", format='%(asctime)s %(message)s', filemode='w', level=logging.INFO)

    global onnx_model, end_names, split_layers
    #Load the onnx model and extract the final output names
    onnx_model = onnx.load(onnx_file)
    end_names = []
    for i in range(len(onnx_model.graph.output)):
        end_names.append(onnx_model.graph.output[i].name)

    #Load the list of the possible split points
    #with open("temp/split_layers", "rb") as fp:   # Unpickling
    #    split_layers = pickle.load(fp)
    split_layers = []
 
    @app.route("/split_layers", methods=["POST", "GET"])
    def get_split_layers():
       global split_layers
       split_layers = request.json["split_layers"]
       return {"Outcome": "Success!"}

    @app.route("/")
    def root():
        return "<h1>Hello There</h1>"

    @app.route("/status")
    def return_server_status():
        return "<h1>Operational</h1>"

    @app.route("/endpoint", methods=["POST", "GET"])
    def endpoint():
        global onnx_model, end_names, split_layers

        #Receive the incoming data
        data = json.load(request.files['data'])
        data["result"] = np.load(request.files['document']).tolist()

        #Compute arrival time 
        arrival_time = time.time()

        print("###### Input layer: %s ######" %data["splitLayer"])
        #Split the model to obtain the third submodel
        if data["splitLayer"] == "NO_SPLIT":
            input_names = onnx_get_true_inputs(onnx_model)
            onnx_model_file = "temp/endpoint_no_split.onnx"
        else:
            input_names = [data["splitLayer"]]
            input_layer_index = split_layers.index(data["splitLayer"])
            onnx_model_file = "temp/endpoint_" + str(input_layer_index) + ".onnx"

        output_names = end_names
        
        if os.path.isfile(onnx_model_file):
            print("Subomdel already extracted!")
        else:
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
        global onnx_model, end_names, split_layers

        #Receive the incoming data
        data = json.load(request.files['data'])
        data["result"] = np.load(request.files['document']).tolist()

        #Compute arrival time 
        arrival_time = time.time()

        #Extract the input layer and its index into the list of possible splits
        if data["splitLayer"] == "NO_SPLIT":
            input_names = onnx_get_true_inputs(onnx_model)
            input_layer_index = -1
        else:
            input_names = [data["splitLayer"]]
            input_layer_index = split_layers.index(data["splitLayer"])

        onnx_model_file = "temp/second_half.onnx"

        results_file = data["splitLayer"].replace("/", '-').replace(":", '_') + ".csv"
        with open(results_file, "a", newline="") as csvfile:
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
                    #We use this simple trick to "fool" onnx_search_and_run_second_half
                    in_data = data
                    in_data["splitLayer"] = ""
                    #Split the model to obtain the second submodel and compute the time needed to run it
                    try:
                        up_data = onnx_extract_and_run_second_half(onnx_file, input_names, output_names, 
                                                                onnx_model_file, in_data, None, EP_list, device)
                    except:
                       print("...CANNOT EXTRACT AND RUN THE SUBMODEL!...")
                       raise Exception("EXTRACTION FAILED")
                    
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
                    try:
                        response = requests.post(server_url, files=files).json() 

                        #Save the results
                        row["splitPoint2"] = split_layers[i]
                        row["execTime1"] = response["execTime1"]
                        row["execTime2"] = response["execTime2"] 
                        row["networkingTime"] = response["arrival_time"] - departure_time
                        writer.writerow(row)  
                    except:
                        print("...ENDPOINT FAILED!...")
                        raise Exception("ENDPOINT FAILED") 
                    
                    logging.info(str(input_layer_index) + " to " + str(i) + ": OK")           
                except Exception as e:
                    error_message = str(e)
                    print(error_message)
                    logging.error(str(input_layer_index) + " to " + str(i) + ": " + error_message)   
            
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

