from flask import Flask, request
import requests
import time
from onnx_second_inference_flask import onnx_search_and_run_second_half, onnx_extract_and_run_second_half
import numpy as np
import json
import onnx
import logging
import csv
import os, shutil
import click
from onnx_helper import CsvHandler


def isNodeAnInitializer(onnx_model, node):
    """
    Check if the node passed as argument is an initializer in the network.

    :param onnx_model: the already imported ONNX Model
    :param node: node's name
    :returns: True if the node is an initializer, False otherwise
    """
    # Check if input is not an initializer, if so ignore it
    for i in range(len(onnx_model.graph.initializer)):
        if node == onnx_model.graph.initializer[i].name:
            return True

    return False


def onnx_get_true_inputs(onnx_model):
    """
    Get the list of TRUE inputs of the ONNX model passed as argument.
    The reason for this is that sometimes "onnx.load" interprets some of the static initializers
    (such as weights and biases) as inputs, therefore showing a large list of inputs and misleading for instance
    the fuctions used for splitting.

    :param onnx_model: the already imported ONNX Model
    :returns: a list of the true inputs
    """
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
@click.option(
    "--server_url",
    default="",
    help="Set the url of the next device on the chain. If this is the last layer, use an empyt string",
)
@click.option("--log_file", default="", help="Select where to save the log of the operation performed")
@click.option(
    "--exec_provider",
    default="CPU",
    help="Select the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)",
)
@click.option(
    "--device", default=None, help="Specify the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc.."
)
@click.option("--threshold", default=10.0, help="Specify the threshold above which we skip the iteration")
@click.option("--port", default=5000, help="Select the port where to run the flask app")
@click.option("--host", default="127.0.0.1", help="Select where to host the flask app")
def main(onnx_file, server_url, log_file, exec_provider, device, threshold, port, host):
    if os.path.isdir("cache"):
        shutil.rmtree("cache")
    if exec_provider == "GPU":
        EP_list = ["CUDAExecutionProvider"]
    else:
        EP_list = ["CPUExecutionProvider"]
    app = run(onnx_file, server_url, log_file, EP_list, device, threshold)
    app.run(port=port, host=host)


def run(onnx_file, server_url, log_file, EP_list, device, threshold):
    """
    A flask application factory for extract and run slices of an onnx model at inference on multiple devices

    :param onnx_file: the ONNX file to use for the inference
    :param server_url: specifies the url of the next device on the chain. If this is the last layer, use an empty string
    :param log_file: specifies the file where to save the log of the operation performed
    :param EP_list: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
    :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
    :param threshold: the threshold on the networking time above which we skip inference
    :return: the blueprint for a flask app which can be run by calling the method run
    """
    app = Flask(__name__)

    # Configure the logger
    logging.basicConfig(filename=log_file, format="%(asctime)s %(message)s", filemode="w", level=logging.INFO)

    global onnx_model, end_names, split_layers, nextDev_times, is_linkingend

    # Sync the clock of the device with the NTP servers
    # if sync_time:
    #    c = ntplib.NTPClient()
    #    response = c.request('ntp1.inrim.it')
    #    offset = response.offset
    #    delay = response.delay
    #    correction = offset - delay/2
    # else:
    #    correction = 0.

    # Load the onnx model and extract the final output names
    onnx_model = onnx.load(onnx_file)
    end_names = []
    for i in range(len(onnx_model.graph.output)):
        end_names.append(onnx_model.graph.output[i].name)

    split_layers = []
    nextDev_times = {}
    is_linkingend = False

    # Check if cache directory exists, otherwise create it
    cache_directory_path = "cache"
    if not os.path.isdir(cache_directory_path):
        os.makedirs(cache_directory_path)

    @app.before_request
    def before_request():
        global request_start_time
        request_start_time = time.time()

    @app.route("/position", methods=["GET"])
    def get_my_position():
        """
        Get the position of the device in the chain so that it knows if it's linking to the last server layer or to another
        intermediate server layer
        """
        global is_linkingend
        if server_url == "":
            print("Last layer reached!")
            return {"next": "last_layer"}
        else:
            url = server_url + "/position"
            print("Uploading to %s" % url)
            response = requests.get(url).json()
            if response["next"] == "last_layer":
                print("Linking to last layer...")
                is_linkingend = True
            else:
                print("Linking to an intermediate layer...")
            return {"next": "intermediate_layer"}

    @app.route("/split_layers", methods=["POST", "GET"])
    def get_split_layers():
        """
        Get the list with the split points in the onnx model from the previous device and send it to the next
        device in the chain
        """
        global split_layers
        split_layers = request.json["split_layers"]
        if server_url == "":  # If we have reached the end of the chain stop
            print("Last layer reached!")
            return {"Outcome": "Success!"}
        else:  # Else send the list of the split points to the next device
            url = server_url + "/split_layers"
            print("Uploading to %s" % url)
            response = requests.post(url, json={"split_layers": split_layers}).json()
            return {"Outcome": response["Outcome"]}

    @app.route("/next_iteration", methods=["GET"])
    def clear_cached_times():
        """
        Clear the eventual cached times and tell the next device to do so
        """
        global nextDev_times
        nextDev_times = {}
        if server_url == "":  # If we have reached the end of the chain stop
            print("Last layer reached!")
            return {"Outcome": "Cached times cleared!"}
        else:  # Else proceed recursively till we reach the end
            url = server_url + "/next_iteration"
            print("Uploading to %s" % url)
            response = requests.get(url).json()
            return {"Outcome": response["Outcome"]}

    @app.route("/")
    def root():
        return "<h1>Hello There</h1>"

    @app.route("/status")
    def return_server_status():
        return "<h1>Operational</h1>"

    @app.route("/last_layer", methods=["POST", "GET"])
    def last_layer():
        """
        Extract and run at inference the last submodel of the partition of a onnx model
        """
        global onnx_model, end_names, split_layers, request_start_time

        # Receive the incoming data
        data = json.load(request.files["data"])
        data["result"] = np.load(request.files["document"]).tolist()

        # Compute arrival time
        arrival_time = time.time()

        # Compute uploading time
        uploading_time = arrival_time - request_start_time

        if uploading_time >= threshold:  # If uploading time is too big we skip the profiling
            return {"Outcome": "Threshold exceeded"}

        print("###### Input layer: %s ######" % data["splitLayer"])
        # Split the model to obtain the third submodel
        if data["splitLayer"] == "NO_SPLIT":
            input_names = onnx_get_true_inputs(onnx_model)
            onnx_model_file = "cache/last_no_split.onnx"
        else:
            input_names = [data["splitLayer"]]
            try:  # Try to find the index of the layer inside the list
                input_layer_index = split_layers.index(data["splitLayer"])
                onnx_model_file = "cache/last_" + str(input_layer_index) + ".onnx"
            except:  # Otherwise use the name of the split layer to cache the onnx model
                onnx_model_file = "cache/last_" + data["splitLayer"].replace("/", "-").replace(":", "_") + ".onnx"

        output_names = end_names

        if os.path.isfile(onnx_model_file):
            print("Subomdel already extracted!")
        else:
            onnx.utils.extract_model(onnx_file, onnx_model_file, input_names, output_names)

        # Compute the time needed to run the third submodel
        returnData = onnx_search_and_run_second_half(None, onnx_model_file, data, None, EP_list, device)

        # Return the results
        returnData["Outcome"] = "Success"
        returnData["networkingTime"] = uploading_time
        returnData["result"] = returnData["result"].tolist()
        return returnData

    @app.route("/intermediate_layer", methods=["POST", "GET"])
    def intermediate_layer():
        """
        Extract and run at inference all the possible submodel extracted from a onnx model starting from a certain
        split point. Send the output of the runs to the next device and get the results.
        """
        global onnx_model, end_names, split_layers, nextDev_times, is_linkingend, request_start_time

        # Receive the incoming data
        data = json.load(request.files["data"])
        data["result"] = np.load(request.files["document"]).tolist()

        # Compute arrival time
        arrival_time = time.time()

        # Extract the input layer and its index into the list of possible splits
        if data["splitLayer"] == "NO_SPLIT":
            input_names = onnx_get_true_inputs(onnx_model)
            input_layer_index = -1
        else:
            input_names = [data["splitLayer"]]
            input_layer_index = split_layers.index(data["splitLayer"])

        results_file = data["splitLayer"].replace("/", "-").replace(":", "_") + ".csv"
        with open(results_file, "a", newline="") as csvfile:
            fields = [
                "SplitLayer",
                "1stInfTime",
                "2ndInfTime",
                "networkingTime",
                # "tensorSaveTime",
                "tensorLoadTime",
                "tensorLength",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            row = {
                "SplitLayer": "",
                "1stInfTime": 0.0,
                "2ndInfTime": 0.0,
                "networkingTime": 0.0,
                # "tensorSaveTime": 0.0,
                "tensorLoadTime": 0.0,
                "tensorLength": 0,
            }

            if arrival_time - data["departure_time"] < threshold:  # If uploading time is too big we skip the profiling
                for i in range(input_layer_index + 1, len(split_layers)):
                    # If the uploading time is too big, we skip this iteration
                    if split_layers[i] in nextDev_times:
                        response = nextDev_times[split_layers[i]]
                        if response["Outcome"] == "Threshold exceeded":
                            logging.info(str(input_layer_index) + " to " + str(i) + ": THRESHOLD EXCEEDED")
                            continue

                    # Find the second split point
                    name = split_layers[i]
                    output_layer_index = split_layers.index(name)
                    print("##### Output layer: %s #####" % name)
                    output_names = [name]

                    # Compute the name of the file where we will export the submodel
                    if input_layer_index >= 0:
                        onnx_model_file = (
                            "cache/intermediate_" + str(input_layer_index) + "_" + str(output_layer_index) + ".onnx"
                        )
                    else:
                        onnx_model_file = "cache/intermediate_no_split_" + str(output_layer_index) + ".onnx"

                    try:
                        # We use this simple trick to "fool" onnx_search_and_run_second_half
                        data["splitLayer"] = ""
                        # Split the model to obtain the second submodel and compute the time needed to run it
                        try:
                            up_data = onnx_extract_and_run_second_half(
                                onnx_file, input_names, output_names, onnx_model_file, data, None, EP_list, device
                            )
                        except Exception as e:
                            print("...CANNOT EXTRACT AND RUN THE SUBMODEL!...")
                            raise e

                        up_data["tensorLength"] = up_data["result"].size
                        np.save("input_check", up_data["result"])
                        del up_data["result"]
                        up_data["splitLayer"] = name
                        up_data["execTime1"] = up_data["execTime2"]
                        # Embed departure time inside uploading data
                        departure_time = time.time()
                        up_data["departure_time"] = departure_time
                        files = [
                            ("document", ("input_check.npy", open("input_check.npy", "rb"), "application/octet")),
                            ("data", ("up_data", json.dumps(up_data), "application/json")),
                        ]

                        if not split_layers[i] in nextDev_times:
                            # Send the Intermediate Tensors to the server
                            print("Sending the intermediate tensors to the server...")
                            try:
                                # Choose the correct url depending if we are linkging to the last server layer or to a second intermediate layer
                                if is_linkingend:
                                    url = server_url + "/last_layer"
                                else:
                                    url = server_url + "/intermediate_layer"
                                response = requests.post(url, files=files).json()

                                # Save the inference time of the submodel on the next device for future iterations
                                nextDev_times[split_layers[i]] = response

                                if response["Outcome"] != "Threshold exceeded":
                                    # Save the results
                                    row["SplitLayer"] = split_layers[i].replace("/", "-").replace(":", "_")
                                    row["1stInfTime"] = response["execTime1"]
                                    row["2ndInfTime"] = response["execTime2"]
                                    row["networkingTime"] = response["networkingTime"]
                                    # row["tensorSaveTime"] = response["tensorSaveTime"]
                                    row["tensorLoadTime"] = response["tensorLoadTime"]
                                    row["tensorLength"] = response["tensorLength"]
                                    writer.writerow(row)
                                    logging.info(str(input_layer_index) + " to " + str(i) + ": OK")
                                else:
                                    logging.info(str(input_layer_index) + " to " + str(i) + ": THRESHOLD EXCEEDED")
                            except:
                                print("...LAST LAYER FAILED!...")
                                raise Exception("LAST LAYER FAILED")
                        else:
                            response = nextDev_times[split_layers[i]]
                            # Save the results
                            row["SplitLayer"] = split_layers[i].replace("/", "-").replace(":", "_")
                            row["1stInfTime"] = up_data["execTime1"]
                            row["2ndInfTime"] = response["execTime2"]
                            row["networkingTime"] = response["networkingTime"]
                            # row["tensorSaveTime"] = response["tensorSaveTime"]
                            row["tensorLoadTime"] = response["tensorLoadTime"]
                            row["tensorLength"] = response["tensorLength"]
                            writer.writerow(row)
                            logging.info(str(input_layer_index) + " to " + str(i) + ": OK")

                    except Exception as e:
                        error_message = str(e)
                        print(error_message)
                        logging.error(str(input_layer_index) + " to " + str(i) + ": " + error_message)
            else:
                print("Networking time is too big, skipping the iteration...")

            # Trivial case: we don't rely on the server
            print("##### Trivial case #####")
            output_names = end_names
            onnx_model_file = "cache/intermediate_" + str(input_layer_index) + "_no_split.onnx"
            returnData = onnx_extract_and_run_second_half(
                onnx_file, input_names, output_names, onnx_model_file, data, None, EP_list, device
            )

            # Save the results
            row["SplitLayer"] = "NO_SPLIT"
            row["1stInfTime"] = returnData["execTime2"]
            row["2ndInfTime"] = 0.0
            row["networkingTime"] = 0.0
            # row["tensorSaveTime"] = 0.0
            row["tensorLoadTime"] = 0.0
            row["tensorLength"] = 0
            writer.writerow(row)

            logging.info(str(input_layer_index) + " to end: OK")

            # Return the results
            returnData["Outcome"] = "Success"
            returnData["networkingTime"] = arrival_time - request_start_time
            returnData["result"] = returnData["result"].tolist()

        return returnData

    @app.route("/end", methods=["GET"])
    def finalize():
        """
        At the end of all repetitions, generate the files with the average times
        """
        global split_layers
        if server_url == "":  # If we have reached the end of the chain stop
            print("Last layer reached!")
            return {"Outcome": "Success!"}
        else:  # Else compute the avgs for all the possible split layers
            # Extract the name of the files
            file_names = split_layers.copy()
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace("/", "-").replace(":", "_")
            # Append NO_SPLIT
            file_names.append("NO_SPLIT")
            for i in range(len(file_names)):  # For every possible .csv file
                csvfile = file_names[i] + ".csv"
                if os.path.isfile(csvfile):  # If the file does exist
                    handler = CsvHandler(csvfile)
                    handler.export_mean_values()
                    handler.reorder(file_names)
            # Proceed recursively along the chain
            url = server_url + "/end"
            print("Uploading to %s" % url)
            response = requests.get(url).json()
            return {"Outcome": response["Outcome"]}

    return app


if __name__ == "__main__":
    main()
