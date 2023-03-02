# **ONNX-PROFILER**
## **Introduction**
onnx-profiler is a tool for partitioning `.onnx` model and profiling them by running the partitioned models on multiple devices. Born as a fork of [onnx-splitter](https://github.com/thestarivore/onnx-splitter), onnx-profiler:
- replaces the original partitioner with a new more performing one
- replaces OSCAR with an http request to a flask application
- allows to work with three or more devices.
## **Usage example**
A quick example where we are going to partition the model `mobilenet.onnx` and run the partitioned model on three devices. For this quick example we will use a single device and we will emulate the second and the third ones by running them on port 5000 and 3000. 
### **1. Split the model**
```
python onnx_cutter.py --onnx_model=mobilenet.onnx --ouput_path=onnx_models/mobilenet --num_partitions=10 --pickle_file=temp/split_layers
```
We find the 10 best points where to split `mobilenet.onnx` and we save the list with them in the file `temp/split_layers`. We also place inside `onnx_models/mobilnet` an initial partitioning of the model.
### **2. Turn on the server on the third device**
```
python onnx_server.py --onnx_file=mobilenet.onnx --log_file=endpoint.log --port=3000
```
By not passing the optional argument `--server_url`, the third device knows to be the *endpoint*.

### **3. Turn on the server on the second device**
```
python onnx_server.py --onnx_file=mobilenet.onnx --log_file=checkpoint.log --server_url=https://127.0.0.1:3000
```
By passing a valid `server_url`, the second device knows to be a *checkpoint*.

### **4. Run the tool on the client**
```
python onnx_manager_flask.py --operation run_all --onnx_file mobilenet.onnx --onnx_path onnx_models/mobilenet --image_batch images/mobilenet --image_size_x 160 --image_size_y 160 --image_is_grayscale False --server_url http://127.0.0.1:5000/checkpoint --pickle_file temp/split_layers
```
To have a better insight on how to pass additional parameters to the tool run 
```
python onnx_manager_flask.py --help
```
or 
```
python onnx_server.py --help
```
and have a look at the documentation of the original tool [here](https://github.com/thestarivore/onnx-splitter).
