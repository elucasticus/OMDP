import numpy as np
import requests
import time
from onnx_manager_flask import onnx_run_complete, onnx_run_all_complete, onnx_run_profiler

def main():
    # x = np.random.randn(3, 1).flatten()
    # data = {'x': x.tolist()}
    # departure_time = time.time()
    # response = requests.post("http://127.0.0.1:5000/test", json=data).json()
    # arrival_time = response["arrival_time"]
    # uploading_time = arrival_time - departure_time
    # print(uploading_time)
    onnx_file = "mobilenet.onnx"
    onnx_path = "onnx_models/mobilenet"
    split_layer = "sequential/mobilenetv2_1.00_160/block_13_project_BN/FusedBatchNormV3:0"
    split_layer = split_layer.replace("/", '-').replace(":", '_')
    onnx_run_complete(onnx_file, "NO_SPLIT", None, "images/mobilenet", 160, 160, False , "CPU", "CPU", "http://127.0.0.1:5000/checkpoint")


if __name__ == "__main__":
    main()


