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

    onnx_file = "cifar10.onnx"
    onnx_path = "cifar10"
    onnx_run_all_complete(onnx_file, onnx_path, None, "images", 32, 32, False, 2, "CPU", None, "http://127.0.0.1:5000/onnx")

if __name__ == "__main__":
    main()


