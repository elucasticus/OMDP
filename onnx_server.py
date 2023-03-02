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
import os, shutil
import click

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

@click.command()
@click.option("--onnx_file", help="Select the ONNX file to use for the inference")
@click.option("--server_url", default="", help="Set the url of the next device on the chain. If this is the endpoint, use an empyt string")
@click.option("--log_file", default="", help="Select where to save the log of the operation performed")
@click.option("--EP_list", "EP_list", default="CPU", help="Select the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)")
@click.option("--device", default=None, help="Specify the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..")
@click.option("--threshold", default=10., help="Specify the threshold above which we skip the iteration")
@click.option("--port", default=5000, help="Select the port where to run the flask app")
@click.option("--host", default="127.0.0.1", help="Select where to host the flask app")
def main(onnx_file, server_url, log_file, EP_list, device, threshold, port, host):
    shutil.rmtree("cache")
    if EP_list == "CPU":
       EP_list = ["CPUExecutionProvider"]
    app = run(onnx_file, server_url, log_file, EP_list, device, threshold)
    app.run(port=port, host=host)

def run(onnx_file, server_url, log_file, EP_list, device, threshold):
    '''
    Use an input model OR search the correct one and run at inference the Second Half of the Splitted Model

    :param onnx_file: the ONNX file to use for the inference
    :param server_url: specifies the url of the next device on the chain. If this is the endpoint, use an empyt string
    :param log_file: specifies the file where to save the log of the operation performed
    :param EP_list: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
    :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
    '''
    app = Flask(__name__)

    #Configure the logger 
    logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s', filemode='w', level=logging.INFO)

    global onnx_model, end_names, split_layers, nextDev_times

    #Load the onnx model and extract the final output names
    onnx_model = onnx.load(onnx_file)
    end_names = []
    for i in range(len(onnx_model.graph.output)):
        end_names.append(onnx_model.graph.output[i].name)

    split_layers = []
    nextDev_times = {}

    #Check if cache directory exists, otherwise create it
    cache_directory_path = "cache"
    if not os.path.isdir(cache_directory_path):
       os.makedirs(cache_directory_path)
 
    @app.route("/split_layers", methods=["POST", "GET"])
    def get_split_layers():
       global split_layers
       split_layers = request.json["split_layers"]
       if server_url == "":             #If we have reached the end of the chain stop
        print("Endpoint reached!")
        return {"Outcome": "Success!"}
       else:                            #Else send the list of the split points to the next device
        url = server_url + "/split_layers"
        print("Uploading to %s" %url)
        response = requests.post(url, json={"split_layers": split_layers}).json()
        return {"Outcome": response["Outcome"]}
       
    @app.route("/next_iteration", methods=["GET"])
    def clear_cached_times():
       global nextDev_times
       nextDev_times = {}
       if server_url == "":             #If we have reached the end of the chain stop
        print("Endpoint reached!")
        return {"Outcome": "Cached times cleared!"}
       else:                            #Else proceed recursively till we reach the end
        url = server_url + "/next_iteration"
        print("Uploading to %s" %url)
        response = requests.get(url).json()
        return {"Outcome": response["Outcome"]}

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
            onnx_model_file = "cache/endpoint_no_split.onnx"
        else:
            input_names = [data["splitLayer"]]
            input_layer_index = split_layers.index(data["splitLayer"])
            onnx_model_file = "cache/endpoint_" + str(input_layer_index) + ".onnx"

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
        global onnx_model, end_names, split_layers, nextDev_times

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

        results_file = data["splitLayer"].replace("/", '-').replace(":", '_') + ".csv"
        with open(results_file, "a", newline="") as csvfile:
            fields = ["splitPoint1", "splitPoint2", "1stInfTime", "2ndInfTime", "networkingTime"]
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            row = {"splitPoint1": data["splitLayer"],
                   "splitPoint2": "",
                   "1stInfTime":0.,
                   "2ndInfTime":0.,
                   "networkingTime": 0.}
            
            if arrival_time - data["departure_time"] < threshold:
                for i in range(input_layer_index + 1, len(split_layers)):
                    #Find the second split point 
                    name = split_layers[i]
                    output_layer_index = split_layers.index(name)
                    print("##### Output layer: %s #####" %name)
                    output_names = [name]

                    #Compute the name of the file where we will export the submodel
                    if input_layer_index >= 0:
                        onnx_model_file = "cache/checkpoint_" + str(input_layer_index) + "_" + str(output_layer_index) + ".onnx"
                    else:
                        onnx_model_file = "cache/checkpoint_no_split_" + str(output_layer_index) + ".onnx"
                    
                    try:
                        #We use this simple trick to "fool" onnx_search_and_run_second_half
                        in_data = data
                        in_data["splitLayer"] = ""
                        #Split the model to obtain the second submodel and compute the time needed to run it
                        try:
                            up_data = onnx_extract_and_run_second_half(onnx_file, input_names, output_names, 
                                                                    onnx_model_file, in_data, None, EP_list, device)
                        except Exception as e:
                            print("...CANNOT EXTRACT AND RUN THE SUBMODEL!...")
                            raise e
                        
                        np.save("input_check", up_data["result"])
                        del up_data["result"]
                        up_data["splitLayer"] = name
                        up_data["execTime1"] = up_data["execTime2"]
                        files = [
                            ('document', ("input_check.npy", open("input_check.npy", 'rb'), 'application/octet')),
                            ('data', ('up_data', json.dumps(up_data), 'application/json')),
                        ]

                        if not split_layers[i] in nextDev_times:
                            #Send the Intermediate Tensors to the server
                            print("Sending the intermediate tensors to the server...")
                            departure_time = time.time()
                            try:
                                url = server_url + "/endpoint"
                                response = requests.post(url, files=files).json()
                                response["networkingTime"] = response["arrival_time"] - departure_time

                                #Save the inference time of the submodel on the next device for future iterations
                                nextDev_times[split_layers[i]] = response

                                #Save the results
                                row["splitPoint2"] = split_layers[i]
                                row["1stInfTime"] = response["execTime1"]
                                row["2ndInfTime"] = response["execTime2"] 
                                row["networkingTime"] = response["networkingTime"]
                                writer.writerow(row)  
                            except:
                                print("...ENDPOINT FAILED!...")
                                raise Exception("ENDPOINT FAILED")
                        else:
                            response = nextDev_times[split_layers[i]]
                            #Save the results
                            row["splitPoint2"] = split_layers[i]
                            row["1stInfTime"] = up_data["execTime1"]
                            row["2ndInfTime"] = response["execTime2"] 
                            row["networkingTime"] = response["networkingTime"]
                            writer.writerow(row) 
                            
                        logging.info(str(input_layer_index) + " to " + str(i) + ": OK")           
                    except Exception as e:
                        error_message = str(e)
                        print(error_message)
                        logging.error(str(input_layer_index) + " to " + str(i) + ": " + error_message)
            else:
               print("Networking time is too big, skipping the iteration...")   
                
            #Trivial case: we don't rely on the server
            print("##### Trivial case #####")
            output_names = end_names
            onnx_model_file = "cache/checkpoint_" + str(input_layer_index) +"_no_split.onnx"
            returnData = onnx_extract_and_run_second_half(onnx_file, input_names, output_names, 
                                                            onnx_model_file, data, None, EP_list, device)

            #Save the results 
            row["splitPoint2"] = "end"
            row["1stInfTime"] = returnData["execTime2"]
            row["2ndInfTime"] = 0.
            row["networkingTime"] = 0.
            writer.writerow(row)

            logging.info(str(input_layer_index) + " to end: OK")

            #Return the results
            returnData["Outcome"] = "Success"
            returnData["arrival_time"] = arrival_time
            returnData["result"] = returnData["result"].tolist()

        return returnData
    
    return app

if __name__ == "__main__":
    main()
