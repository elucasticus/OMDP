from numpy.core.numeric import True_
#import sclblonnx as so
import onnx
#from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
#from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import datetime
import numpy as np
import argparse
import time
import json
import os
import csv
from onnx_first_inference import onnx_run_first_half
import pickle
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import onnxruntime
#from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantFormat, QuantType
from onnx_opcounter import calculate_params, calculate_macs
from onnx2json import convert
from onnx_splitter import (onnx_model_split, onnx_model_split_all, onnx_model_multi_split, onnx_model_early_exits_split, 
                           onnx_model_split_all_singlenode, onnx_show_graph)
from onnx_utils import *
from packaging import version
import copy
import onnx_tool
from onnx_tool import create_ndarray_f32

#Dependencies for flask
import requests

# Prefer ACL Execution Provider over CPU Execution Provider
ACL_EP_list       = ['ACLExecutionProvider']
# Prefer OpenVINO Execution Provider over CPU Execution Provider
TensorRT_EP_list  = ['TensorrtExecutionProvider']
# Prefer OpenVINO Execution Provider over CPU Execution Provider
OpenVINO_EP_list  = ['OpenVINOExecutionProvider']
# Prefer CUDA Execution Provider over CPU Execution Provider
GPU_EP_list       = ['CUDAExecutionProvider']
# Prefer CPU Execution Provider
CPU_EP_list       = ['CPUExecutionProvider']

# TESTING
dictTensors = {} 

def main():
  '''
    Manages operations on ONNX DNN Models such as layer visualizzation and splitting.

    Arguments:
    -h, --help            Show this help message and exit
    --operation OPERATION
                          Select the operation to be performed on the ONNX Model
                          (list_layers | print_model | split_model | split_model_all | multi_split_model | early_exit_split_model | data_processing | 
                           run | run_all | run_profiler | plot_results | quant_model | show_graph)
    --onnx_file ONNX_FILE                       Select the ONNX File
    --split_layer SPLIT_LAYER                   Select the layer where the slit must take place on the ONNX Model
    --split_layers SPLIT_LAYERS                 Select the list of layers where the slit must take place on the ONNX Model
    --outputs OUTPUTS                           Select the output and the early exits where the slit must take place on the ONNX Model (the actual split will take place above the early exit)
    --onnx_path ONNX_PATH                       Select the path were all the Splitted ONNX Models are stored
    --image_file IMAGE_FILE                     Select the Image File
    --image_file IMAGE_BATCH                    Select the Image Folder containing the Batch of images
    --image_size_x IMAGE_SIZE_X                 Select the Image Size X
    --image_size_y IMAGE_SIZE_Y                 Select the Image Size Y
    --image_is_grayscale IMAGE_IS_GRAYSCALE     Indicate if the Image is in grayscale
    --results_file RESULTS_FILE                 Select the Results File(.csv)
    --quant_type QUANT_TYPE                     Choose weight type used during model quantization
                                                Dependencies: https://www.tensorflow.org/install/source#gpu
    --device_type Device_TYPE   Select DeviceType
    --exec_type EXEC_TYPE                       Select Execution Provider at inference: CPU (default) | GPU | OpenVINO | TensorRT | ACL
                                'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16', 'VAD-F_FP32',
                                Options are: (Any hardware target can be assigned if you have the access to it)
                                'HETERO:MYRIAD,CPU',  'MULTI:MYRIAD,GPU,CPU'
    --rep REP                                   Number of repetitions
  '''
  parser=argparse.ArgumentParser(
    description='''
ONNX Manager: Manages operations on ONNX DNN Models such as: 
  - the execution of the complete cycle edge-cloud;
  - the creation of splitted models;
  - model layer visualizzation;
  - data processing (of images or batches); 
  - the quantization of models;
  - plotting of results;
  - showing the graph on ONNX Models
    ''',
    epilog='''
Examples:
> python onnx_manager_flask.py --operation run --split_layer sequential/dense_1/MatMul:0 
                               --onnx_path LENET_SplittedModels/ --image_file=images/mnist_test.jpg 
                               --image_size_x=32 --image_size_y=32 --image_is_grayscale=True
> python onnx_manager_flask.py --operation run_all --onnx_file lenet.onnx --onnx_path LENET_SplittedModels/ 
                               --image_file=images/mnist_test.jpg --image_size_x=32 --image_size_y=32 --image_is_grayscale=True

> python onnx_manager_flask.py --operation list_layers --onnx_file mobilenet_v2.onnx
> python onnx_manager_flask.py --operation split_model --onnx_file mobilenet_v2.onnx 
                               --split_layer sequential/mobilenetv2_1.00_160/block_3_project_BN/FusedBatchNormV3:0

> python onnx_manager_flask.py --operation run --split_layer sequential/mobilenetv2_1.00_160/block_5_add/add:0 
                               --onnx_path MobileNetV2_SplittedModles 
                               --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
                               --image_size_x=160 --image_size_y=160 --image_is_grayscale=False --minio_bucket=onnx-test-mobilenet 
                               --oscar_service=onnx-test-mobilenet
> python onnx_manager_flask.py --operation run --split_layer sequential/mobilenetv2_1.00_160/block_5_add/add:0 
                               --onnx_path MobileNetV2_SplittedModles --image_batch=images/mobilenet_batch 
                               --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
                               --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet

> python onnx_manager_flask.py --operation run_all --onnx_file mobilenet_v2.onnx --onnx_path MobileNetV2_SplittedModles 
                               --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
                               --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
                               --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet
> python onnx_manager_flask.py --operation run_all --onnx_file mobilenet_v2.onnx --onnx_path MobileNetV2_SplittedModles 
                               --image_file=images/mobilenet_batch --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
                               --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet

> python onnx_manager_flask.py --operation data_processing --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
                               --image_size_x=160 --image_size_y=160 --image_is_grayscale=False
    ''',
    formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument('--operation', help='Select the operation to be performed on the ONNX Model',
                      choices=['list_layers', 'print_model', 'split_model', 'split_model_all', 'multi_split_model', 'early_exit_split_model',
                               'data_processing', 'run', 'run_all', 'run_profiler', 'plot_results', 'quant_model', 'show_graph',
                               'prep_ml_dataset', 'prediction_profiling'])
  parser.add_argument('--onnx_file', help='Select the ONNX File')
  parser.add_argument('--xml_file', help='Select the XML File (OpenVINO Optimized Model)')
  parser.add_argument('--split_layer', help='Select the layer where the slit must take place on the ONNX Model')
  parser.add_argument('--split_layers', help='Select the list of layers where the slit must take place on the ONNX Model', 
                                        dest='split_layers', type=str, nargs='+')
  parser.add_argument('--outputs',  help='Select the output and the early exits where the slit must take place on the ONNX Model (the actual split will take place above the early exit)', 
                                    dest='outputs', type=str, nargs='+')
  parser.add_argument('--onnx_path', help='Select the path were all the Splitted ONNX Models are stored')
  parser.add_argument('--image_file', help='Select the Image File')
  parser.add_argument('--image_batch', help='Select the Image Folder containing the Batch of images')
  parser.add_argument('--image_size_x', help='Select the Image Size X')
  parser.add_argument('--image_size_y', help='Select the Image Size Y')
  parser.add_argument('--image_is_grayscale', help='Indicate if the Image is in grayscale')
  parser.add_argument('--results_file', help='Select the Results File(.csv)')
  parser.add_argument('--exec_type', help='Select Execution Provider at inference', choices=['CPU', 'GPU', 'OpenVINO', 'TensorRT', 'ACL'])
  parser.add_argument('--device_type', help='Select DeviceType: (CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16, VAD-F_FP32, \
                                             HETERO:MYRIAD,CPU,  MULTI:MYRIAD,GPU,CPU)')
  parser.add_argument("--rep", help='Number of repetitions', type=int)
  parser.add_argument('--warmup_time', help='Set a Warmup Time[sec] (default: 60=1m), to run before the execution of the RUN ALL Cycle')
  parser.add_argument('--input_csv_file', help='Select the Input File(.csv)')
  parser.add_argument('--input_avg_csv_file', help='Select the Average Input File(.csv)')
  parser.add_argument('--filter_layers', nargs='+', help='Specify an array of LayerNames to be filtered.')
  parser.add_argument('--filter_nr_nodes', help='Specify the maximum Nr of Nodes per Split to be filtered.')
  parser.add_argument('--server_url', help='Select the Address of the Flask Server')
  args=parser.parse_args()
  print ("Operation: " + args.operation)

  #Get Execution Provider
  exec_provider = None
  if args.exec_type == "ACL":
    exec_provider = ACL_EP_list
  if args.exec_type == "TensorRT":
    exec_provider = TensorRT_EP_list
  elif args.exec_type == "OpenVINO":
    exec_provider = OpenVINO_EP_list
  elif args.exec_type == "GPU":
    exec_provider = GPU_EP_list
  else:
    exec_provider = CPU_EP_list

  #Choose the operation
  if args.operation == "list_layers":
      onnx_list_model_layers(args.onnx_file)
  elif args.operation == "print_layers":
      onnx_model_details(args.onnx_file)
  elif args.operation == "split_model":
      onnx_model_split(args.onnx_file, args.split_layer)
  elif args.operation == "split_model_all":
      onnx_model_split_all(args.onnx_file)
  elif args.operation == "multi_split_model":
      onnx_model_multi_split(args.onnx_file, args.split_layers)
  elif args.operation == "early_exit_split_model":
      onnx_model_early_exits_split(args.onnx_file, args.outputs)
  elif args.operation == "data_processing":
      onnx_import_data(args.image_file, 
                            args.image_batch,
                            int(args.image_size_x), 
                            int(args.image_size_y), 
                            args.image_is_grayscale == "True")
  elif args.operation == "run":
      onnx_run_complete(args.onnx_path, 
                        args.split_layer, 
                        args.image_file, 
                        args.image_batch,
                        int(args.image_size_x), 
                        int(args.image_size_y), 
                        args.image_is_grayscale == "True",
                        exec_provider,
                        args.device_type,
                        args.server_url,
                        args.xml_file)
  elif args.operation == "run_all":
      onnx_run_all_complete(args.onnx_file, 
                            args.onnx_path, 
                            args.image_file, 
                            args.image_batch,
                            int(args.image_size_x), 
                            int(args.image_size_y), 
                            args.image_is_grayscale == "True",
                            args.rep,
                            exec_provider,
                            args.device_type,
                            args.server_url,
                            args.xml_file,
                            args.warmup_time)
  elif args.operation == "run_profiler":
      onnx_run_profiler(args.onnx_file, 
                        args.onnx_path, 
                        args.image_file, 
                        args.image_batch,
                        int(args.image_size_x), 
                        int(args.image_size_y), 
                        args.image_is_grayscale == "True",
                        args.rep,
                        exec_provider,
                        args.device_type,
                        args.server_url,
                        args.xml_file)
  elif args.operation == "plot_results":
      plot_results(args.results_file)
  elif args.operation == "quant_model":
      quantize_dynamic(args.onnx_file, 'quant_'+args.onnx_file, weight_type=args.quant_type)
  elif args.operation == "show_graph":
      onnx_show_graph(args.onnx_file)
  elif args.operation == "prep_ml_dataset":
      prepare_ml_dataset(args.input_csv_file, args.results_file, args.filter_layers, args.filter_nr_nodes)
  elif args.operation == "prediction_profiling":
      predictionProfiling(args.input_csv_file, args.input_avg_csv_file, args.results_file)


def onnx_list_model_layers(onnx_file):
  '''
  List all the layers of an ONNX DNN Model

  :param onnx_file: the ONNX file to analize
  '''
  model_onnx = load_onnx_model(onnx_file)
  for out in enumerate_model_node_outputs(model_onnx):
      print(out)

def onnx_model_details(onnx_file):
  '''
  Print the details of an ONNX DNN Model

  :param onnx_file: the ONNX file to analize
  '''
  onnx_model = onnx.load(onnx_file)
  print(onnx_model)

def onnx_run_complete(onnx_path, split_layer, image_file, image_batch, img_size_x, img_size_y, is_grayscale, 
                      exec_provider, device_type, server_url, xml_file=None):
  '''
  Run a complete cycle of inference, meaning run the first half of the model locally, get the results, load them on the cloud, 
  execute the second part of the model on the cloud and get the relative results.

  :param onnx_path: the path to the collection of models were to find the correct one to use for the inference
  :param split_layer: the name of the DNN layer where the split has been performed on the model choosen
  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :param exec_provider: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device_type: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  :return: the 1° inference execution time, the 2° inference execution time, the 2° inference OSCAR execution time and the 2nd inference Kubernetes pod execution time
  '''
  #Default Argument Values
  if is_grayscale == None: is_grayscale = False

  #TESTING
  #dictTensors = {} 
  global dictTensors 

  #Iterate through the subdirectories to find the ONNX model splitted at our selected layer
  onnx_file = None
  if split_layer == "NO_SPLIT" or split_layer == "PROFILING": 
    #Since we skip the local execution and don't use splits, the full model is required instead of the onnx_path
    onnx_file = onnx_path
  else:
    split_layer = split_layer.replace("/", '-').replace(":", '_')
    print("Search for: " + split_layer)
    for dir in os.listdir(onnx_path):
      if dir.find("_on_") > 0:
        index = dir.index('_on_')
        d = dir[index+4:]
        #print("Check: " + d)
        if d == split_layer:
          print("Found Layer: " + d)
          onnx_file = onnx_path + "/" + dir + "/first_half.onnx"
          break
  
  # Only proceed if an onnx file is found for this layer
  if onnx_file != None:
    print("\n ###Start the 1st Inference Execution Locally\n")

    #Load the Onnx Model    --    Got to do it just to have the input tensor shape for data_processing
    model_onnx = load_onnx_model(onnx_file)

    # Process input data (image or batch of images)
    inputData = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale, model_onnx.graph.input[0])

    # Check if we have to SKIP the 1st Inference Execution Locally
    if split_layer == "NO_SPLIT" or split_layer == "PROFILING":
      print("\n ###SKIP the 1st Inference Execution Locally, run directly the whole model on the Cloud..\n")
      print(" Create a results.txt file with the whole image instead of the tensor..")
      data = {
        "splitLayer": "NO_SPLIT",
        "fullModelFile": onnx_file,
        "execTime1": 0,   #1st Inference Execution Time
        "result": inputData,
        "tensorLength": inputData.size,
        "tensorSaveTime": 0
      }

      # Profiling case must be differentiated
      if split_layer == "PROFILING":
        data["splitLayer"] = "PROFILING"

      #Save the first input tensor (input)
      dictTensors[model_onnx.graph.input[0].name] = inputData    #"first" won't be recognized, use the input of the model

      # Save the Tensor on a file
      with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(data, f)
    else:
      #Run at Inference the First part of the ONNX DNN Model (on single image OR batch)
      print("onnx_file: ", onnx_file)
      resData, _ = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=True, xml_file=xml_file)       #Now getting time with profiling (20/08/22)
      #I don't need to save the file since it's already saved in onnx_first_inference.py 

      #Save the Intermediate Tensors
      print("Saving the intermediate tensor...")
      dictTensors[resData["splitLayer"]] = resData["result"]
      data = resData
    
    np.save("input", data["result"])
    del data["result"]
    files = [
        ('document', ("input.npy", open("input.npy", 'rb'), 'application/octet')),
        ('data', ('data', json.dumps(data), 'application/json')),
    ]
    #Send the Intermediate Tensors to the server
    print("Sending the intermediate tensors to the server...")
    departure_time = time.time()
    response = requests.post(server_url, files=files).json()

    #Compute uploading time
    arrival_time = response["arrival_time"]
    uploading_time = arrival_time - departure_time
    print("uploading_time: %f" %uploading_time)

    #Print the results
    print("#### 1st Inference Execution Time: " + str(response["execTime1"]) + " s")
    print("#### 2nd Inference Execution Time: " + str(response["execTime2"]) + " s")
    print("---------------------------------------------------")
    print("#### Tensor Length: " + str(response["tensorLength"]))
    print("#### 1st Inf. Tensor Save Time: " + str(response["tensorSaveTime"]) + " s")
    print("#### Networking Time: " + str(uploading_time) + " s")
    print("#### 2nd Inf. Tensor Load Time: " + str(response["tensorLoadTime"]) + " s")

    if split_layer == "PROFILING":
      return response["execTime1"], response["execTime2"], 0, 0, response["tensorLength"], response["tensorSaveTime"], response["tensorLoadTime"], uploading_time, response["profilingTableCloud"]
    else: 
      return response["execTime1"], response["execTime2"], 0, 0, response["tensorLength"], response["tensorSaveTime"], response["tensorLoadTime"], uploading_time
  return -1,-1,-1,-1,-1,-1,-1,-1
    
def onnx_run_all_complete(onnx_file, onnx_path, image_file, image_batch, img_size_x, img_size_y, is_grayscale, 
                          repetitions, exec_provider, 
                          device_type, server_url, xml_file = None, warmupTime = None):
  '''
  Run a complete cycle of inference for every splitted pair of models in the folder passed as argument, save the results in a CSV File and Plot the results.
  To run a complete cycle means to run the first half of the model locally, get the results, load them on the cloud, execute 
  the second part of the model on the cloud and get the results.

  :param onnx_file: the full unsplitted ONNX file (used to gather useful information)
  :param onnx_path: the path to the collection of models were to find the correct one to use for the inference
  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :param repetition: specifies the number of repetitions to execute
  :param exec_provider: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  :param warmupTime: a Warmup Time[sec] (default: 60=1m), to run before the execution of the RUN ALL Cycle
  '''
  #Default Argument Values
  if is_grayscale == None: is_grayscale = False
  if repetitions == None: repetitions = 1
  if warmupTime == None: warmupTime = 0#60

  #TESTING
  global dictTensors
  #dictTensors = {}

  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)
  t_1st_inf, t_2nd_inf, t_oscar_job, t_kube_pod, tensor_lenght, t_tensor_save, t_tensor_load, t_networking = 0,0,0,0,0,0,0,0

  # Process input data (image or batch of images)
  inputData = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale, model_onnx.graph.input[0])
  batchSize = inputData.shape[0]

  # A WARMUP procedure is performed before proceding with the RUN_ALL Cycle
  warmupStart = time.perf_counter()
  while (time.perf_counter() - warmupStart) < int(warmupTime):
    _, _ = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file)

  # Get the Inference Time of each layer (it can be also a sequence of nodes) by analyzing the profiling Table
  '''with open("tensor_dict.pkl", "rb") as tf:
    dictTensors = pickle.load(tf)
  listSingleLayerInfProfiling = getSingleLayerExecutionTimeTable(model_onnx, list(dictTensors.keys()), profilingTable)'''

  #Open an cvs file to save the results
  with open(RESULTS_CSV_FILE, 'w', newline='') as csvfile:
    with open(RESULTS_CSV_FILE2, 'w', newline='') as csvfile2:
      # Repeat the whole cycle the specified number of times
      for rep in range(0, repetitions):
        print("##########   REPETITION #%d   ##########" %(rep + 1))
        #fieldnames = ['SplitLayer', 'Time1', 'Time2', 'Time3', 'Time4']
        fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 
                      'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 'networkingTime']
        cvswriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        cvswriter.writeheader()

        # Process input data (image or batch of images)     
        inputData = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale, model_onnx.graph.input[0])
        batchSize =inputData.shape[0]   

        # Process and get the ProfiligTable by running at inference the full model (A single WARMUP execution is also performed before)  ##used to be outside the repetitions
        _, _ = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=True, xml_file=xml_file)
        resData, profilingTable = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=True, xml_file=xml_file)

        #Execute at inference the whole model locally AND Use profiling (for now disabled)
        t_1st_inf = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file)[0]["execTime1"]  
        print("Finished inference of the whole layer locally..")
        #cvswriter.writerow({'SplitLayer':"NO_SPLIT", "Time1":t_1st_inf, "Time2":0, "Time3":0, "Time4":0})
        cvswriter.writerow({"SplitLayer":"NO_SPLIT", "1stInfTime":t_1st_inf, "2ndInfTime":0, "oscarJobTime":0, "kubePodTime":0, 
                            "tensorSaveTime":0, "tensorLoadTime":0, "tensorLength":0, "networkingTime":0})
        print("Saved results..")  

        #Execute at inference the whole model on the Cloud (OSCAR)
        try:
          (t_1st_inf, t_2nd_inf, t_oscar_job, t_kube_pod, 
            tensor_lenght, t_tensor_save, t_tensor_load, t_networking) = onnx_run_complete(onnx_file,  #it should be onnx_path, but since we skip the local execution and don't use splits, we pass the full model
                                                                                          "NO_SPLIT", 
                                                                                          image_file, 
                                                                                          image_batch, 
                                                                                          img_size_x, 
                                                                                          img_size_y, 
                                                                                          is_grayscale,
                                                                                          exec_provider,
                                                                                          device_type,
                                                                                          server_url,
                                                                                          xml_file=xml_file) 
        except Exception as e:
          print("Error on executin RUN Complete cycle: " + str(e))  

        print("Finished inference of the whole layer on the Cloud (OSCAR)..")
        #cvswriter.writerow({'SplitLayer':"NO_SPLIT", "Time1":0, "Time2":t_2nd_inf, "Time3":t_oscar_job, "Time4":t_kube_pod})
        cvswriter.writerow({"SplitLayer":"NO_SPLIT", "1stInfTime":0, "2ndInfTime":t_2nd_inf, "oscarJobTime":t_oscar_job, "kubePodTime":t_kube_pod,
                            "tensorSaveTime":t_tensor_save, "tensorLoadTime":t_tensor_load, "tensorLength":tensor_lenght, "networkingTime":t_networking})
        print("Saved results..")  

        #Make a split for every layer in the model
        ln = 0
        for layer in enumerate_model_node_outputs(model_onnx):
          #Ignore the first and the last layer
          if layer != list(enumerate_model_node_outputs(model_onnx))[0] and layer != list(enumerate_model_node_outputs(model_onnx))[-1]:
          #if layer != list(enumerate_model_node_outputs(model_onnx))[-1]:
            splitLayer = layer.replace("/", '-').replace(":", '_')

            #TODO:delete this when mobilenet_splittedmodels is updated on OSCAR
            #if splitLayer == "sequential-mobilenetv2_1.00_160-Conv1-Conv2D__7426_0":
            #  splitLayer = "sequential-mobilenetv2_1.00_160-Conv1-Conv2D__6_0"

            print("Splitting at layer: " + splitLayer)

            # Make a complete Inference Run of the whole model by splitting at this particular layer
            print("Run..")
            try:
              (t_1st_inf, t_2nd_inf, t_oscar_job, t_kube_pod, 
                tensor_lenght, t_tensor_save, t_tensor_load, t_networking) = onnx_run_complete(onnx_path, 
                                                                                              splitLayer, 
                                                                                              image_file, 
                                                                                              image_batch, 
                                                                                              img_size_x, 
                                                                                              img_size_y, 
                                                                                              is_grayscale,
                                                                                              exec_provider,
                                                                                              device_type,
                                                                                              server_url,
                                                                                              xml_file=xml_file)
            except Exception as e:
              print("Error on executin RUN Complete cycle: " + str(e)) 

            # t_1st_inf = 1st Inference Execution Time
            # t_2nd_inf = 2nd Inference Execution Time
            # t_oscar_job = 2nd Inf. OSCAR JOB Exec. Time
            # t_kube_pod = 2nd Inf. Kubernetes POD Exec. Time
            # tensor_lenght = 1st Inf. Tensor Length:
            # t_tensor_save = 1st Inf. Tensor Save Time
            # t_tensor_load = 2nd Inf. Tensor Load Time
            if t_1st_inf != -1 and t_2nd_inf != -1 and t_oscar_job != -1 and t_kube_pod != -1:
              print("Finished inference after splitting at layer: " + splitLayer)
              #cvswriter.writerow({'SplitLayer':splitLayer, "Time1":t_1st_inf, "Time2":t_2nd_inf, "Time3":t_oscar_job, "Time4":t_kube_pod})
              cvswriter.writerow({"SplitLayer":splitLayer, "1stInfTime":t_1st_inf, "2ndInfTime":t_2nd_inf, "oscarJobTime":t_oscar_job, "kubePodTime":t_kube_pod,
                                  "tensorSaveTime":t_tensor_save, "tensorLoadTime":t_tensor_load, "tensorLength":tensor_lenght, "networkingTime":t_networking})
              print("Saved results..")

        #Save tensor dictionary 
        with open("tensor_dict.pkl", "wb") as tf:
          pickle.dump(dictTensors,tf)

        #Load tensor dictionary 
        with open("tensor_dict.pkl", "rb") as tf:
          tensors = pickle.load(tf)

        #Get the list of all the layers we need
        #for OLD onnxruntime versions, we need the names of the nodes to be saved differently (attribute name instead of output from the onnx graph)
        splitNodesCompatibility = []   
        for layer, lnode in enumerate_model_node_outputs(model_onnx, add_node = True):
          #Ignore the first and the last layer
          if layer != list(enumerate_model_node_outputs(model_onnx))[0] and layer != list(enumerate_model_node_outputs(model_onnx))[-1]:
            for dir in os.listdir(onnx_path):
              if dir.find("_on_") > 0:
                index = dir.index('_on_')
                d = dir[index+4:]
                if d == splitLayer:
                  splitNodesCompatibility.append(lnode.name)

        # Get the Inference Time of each layer (it can be also a sequence of nodes) by analyzing the profiling Table
        if version.parse(onnxruntime.__version__) < version.parse("1.10.0"):
          #for old onnxruntime versions, we need different names
          dictSingleLayerInfProfiling = getSingleLayerExecutionTimeTable(model_onnx, splitNodesCompatibility, profilingTable)
          #update keys with the node names that we need
          splitNodes = list(dictTensors.keys())
          for i in range(len(splitNodes)):
            try:
              dictSingleLayerInfProfiling[splitNodes[i]] = dictSingleLayerInfProfiling[splitNodesCompatibility[i]]
              del dictSingleLayerInfProfiling[splitNodesCompatibility[i]]
            except:
              dictSingleLayerInfProfiling[splitNodes[i]] = 0
        else:
          dictSingleLayerInfProfiling = getSingleLayerExecutionTimeTable(model_onnx, list(dictTensors.keys()), profilingTable)

        #Iterate through inputs of the graph and get operation types
        dictOperations = {}
        for node in model_onnx.graph.node:
          name = node.output[0]
          op_type = node.op_type
          if name in dictTensors:
            dictOperations[name] = op_type
            #print (op_type)

        #Now we split the layer in single blocks and run them individually in order to get the inference time per layer
        onnx_model_split_all_singlenode(onnx_file, tensors)

        #Run at inference all the single layer models generated and get the inference execution time and the Nr. of Parameters
        layerTime = {}
        dictNrParams = {}
        dictFLOPS = {}
        dictMACS = {}
        dictMEMORY = {}
        for layer in dictTensors:
          #Ignore the first layer since it will be the input
          #if True: #layer != list(dictTensors.keys())[0]:
          if True: #layer not in onnx_get_true_inputs(model_onnx):
            file = "SingleLayerSplits/"+layer.replace("/", '-').replace(":", '_')+'.onnx'

            if os.path.exists(file):
              #Get the Nr. of Parameters  #IT'S NOT ACCURATE
              single_layer_model_onnx = load_onnx_model(file)
              params = calculate_params(single_layer_model_onnx)
              dictNrParams[layer] = params
              #print('Number of params:', params)

              #MACs & Nr. Parameters Profiling of the SingleLayerTime
              modelpath = 'mobilenet_v2.onnx'
              dynamicInputName = single_layer_model_onnx.graph.input[0].name
              dynamicInputShape = dictTensors[layer].shape
              dynamic_inputs= {dynamicInputName: create_ndarray_f32(dynamicInputShape)}
              onnx_tool.model_profile(file, dynamic_inputs, savenode='single_node_table.csv')

              #Get the data from the MACs & Nr. Parameters Profiling and SUM all the values (now directly get the total)
              with open('single_node_table.csv', 'r', newline='') as inner_csvfile:
                reader = csv.reader(inner_csvfile, delimiter=",")
                for i, line in enumerate(reader):
                  if line[0] == "Total":
                    dictMACS[layer] = line[1]
                    dictMEMORY[layer] = line[3]
                    dictNrParams[layer] = line[5]
              print(file + " MACS: " + str(dictMACS[layer]))
              print(file + " MEMORY: " + str(dictMEMORY[layer]))
              print(file + " NrParams: " + str(dictNrParams[layer]))

              #For every model calc FLOPS
              '''onnx_json = convert(
                onnx_graph=single_layer_model_onnx.graph,
              )'''
              
              #FLOPS CALCULATION IS NOT ACCURATE
              onnx_json = convert(
                input_onnx_file_path=file,
                output_json_path="test.json",
                json_indent=2,
              )
              dictNodeFLOPS = calc_flops(onnx_json, batchSize)
              #Sum all the FLOPS of the nodes inside the Single Layer Model
              dictFLOPS[layer] = 0
              for node in dictNodeFLOPS:
                dictFLOPS[layer] = dictFLOPS[layer] + dictNodeFLOPS[node]
              print(file + " FLOPS: " + str(dictFLOPS[layer]))

              #Execute at inference the whole model locally AND Use profiling (for now disabled)
              try:
                #t_inf = onnx_run_first_half(file, dictTensors[layer], False, exec_provider, device_type, profiling=False)["execTime1"] 
                #t_inf = onnx_run_first_half(file, dictTensors[layer], False, CPU_EP_list, device_type, profiling=False)["execTime1"]  
                inputData = dictTensors[layer]
                #first inference is just to get the model loaded into ram
                resData, _ = onnx_run_first_half(file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file)  
                time.sleep(1)
                #sencond inference with the same model is faster (mimicing a case whre we don't have to load the model in ram)
                resData, _ = onnx_run_first_half(file, inputData, True, exec_provider, device_type, True, xml_file, True)  
                t_inf = resData["execTime1"] 
              except Exception as e:
                print(e)
              layerTime[layer] = t_inf
              time.sleep(10) #it's only needed for OpenVINO, cuz NCS2 runs out of memory

        #Save the inf times in a different csv file     
        fieldnames2 = ['SplitLayer', 'singleLayerInfTime', 'OpType', 'NrParameters', 'Memory', 'MACs', 'singleLayerInfTimeProf']
        cvswriter2 = csv.DictWriter(csvfile2, fieldnames=fieldnames2)
        cvswriter2.writeheader()
        cvswriter2.writerow({"SplitLayer":"NO_SPLIT", "singleLayerInfTime":0,"OpType":0, "NrParameters":0, 'Memory':0, 'MACs':0, "singleLayerInfTimeProf":0})
        cvswriter2.writerow({"SplitLayer":"NO_SPLIT", "singleLayerInfTime":0,"OpType":0, "NrParameters":0, 'Memory':0, 'MACs':0, "singleLayerInfTimeProf":0})

        #dictOperations was saved by using the name of the output node for each model, while
        #layerTime, dictNrParams, dictFLOPS were saved by using the name of the input node for each model
        index = 0
        for layer in layerTime:
          #Ignore the last layer name since layerTime was saved by using the name of the input node
          if layer != list(layerTime.keys())[len(list(layerTime.keys()))-1]:
            #Get next layer
            nextLayer = list(layerTime.keys())[index+1]
            try:
              cvswriter2.writerow({"SplitLayer":nextLayer.replace("/", '-').replace(":", '_'), "singleLayerInfTime":layerTime[layer], 
                                  "OpType":dictOperations[nextLayer], "NrParameters":dictNrParams[nextLayer], 'Memory': dictMEMORY[nextLayer], 'MACs':dictMACS[nextLayer],
                                  "singleLayerInfTimeProf":dictSingleLayerInfProfiling[nextLayer]})
            except Exception as e:
              print(e)
            index = index + 1

  #Get the data from the first two cvs files
  list1 = []
  list2 = []
  with open(RESULTS_CSV_FILE, 'r', newline='') as csvfile1:
    reader = csv.reader(csvfile1, delimiter=",")
    for i, line in enumerate(reader):
      list1.append(line)
  with open(RESULTS_CSV_FILE2, 'r', newline='') as csvfile2:
    reader = csv.reader(csvfile2, delimiter=",")
    for i, line in enumerate(reader):
      list2.append(line)

  #Calc the sum of all the time of execution of all the singleLayer models
  singleLayerTimeSum = [0]*repetitions
  rowsPerRepetition = int(len(list2)/repetitions)
  for rep in range(0, repetitions):
    startRow = int(rep*rowsPerRepetition + 1)
    for i in range(startRow,startRow+rowsPerRepetition-1): 
      singleLayerTimeSum[rep] = singleLayerTimeSum[rep] + float(list2[i][1])
    print("Sum of the Single Layer Models time: " + str(singleLayerTimeSum[rep]) + " | rep: " + str(rep))


  '''
  names_list1 : 'SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 'networkingTime'
  names_list2 : 'SplitLayer', 'singleLayerInfTime', 'OpType', 'NrParameters', 'Memory', 'MACs', 'singleLayerInfTimeProf'
  '''

  #AVERAGE all the times
  list1_avg = copy.deepcopy(list1)
  list2_avg = copy.deepcopy(list2)
  #SUM of all the elements
  for i in range(1, rowsPerRepetition): 
    list1_avg[i][1] = float(list1[i][1])   #1stInfTime
    list1_avg[i][2] = float(list1[i][2])   #2ndInfTime
    list1_avg[i][3] = float(list1[i][3])   #oscarJobTime
    list1_avg[i][4] = float(list1[i][4])   #kubePodTime
    list1_avg[i][5] = float(list1[i][5])   #tensorSaveTime
    list1_avg[i][6] = float(list1[i][6])   #tensorLoadTime
    list1_avg[i][8] = float(list1[i][8])   #networkingTime
    #2nd list
    list2_avg[i][1] = float(list2[i][1])   #singleLayerInfTime
    list2_avg[i][6] = float(list2[i][6])   #singleLayerInfTimeProf

    for rep in range(1, repetitions):
      list1_avg[i][1] = float(list1_avg[i][1]) + float(list1[i+rep*(rowsPerRepetition)][1])   #1stInfTime
      list1_avg[i][2] = float(list1_avg[i][2]) + float(list1[i+rep*(rowsPerRepetition)][2])   #2ndInfTime
      list1_avg[i][3] = float(list1_avg[i][3]) + float(list1[i+rep*(rowsPerRepetition)][3])   #oscarJobTime
      list1_avg[i][4] = float(list1_avg[i][4]) + float(list1[i+rep*(rowsPerRepetition)][4])   #kubePodTime
      list1_avg[i][5] = float(list1_avg[i][5]) + float(list1[i+rep*(rowsPerRepetition)][5])   #tensorSaveTime
      list1_avg[i][6] = float(list1_avg[i][6]) + float(list1[i+rep*(rowsPerRepetition)][6])   #tensorLoadTime
      list1_avg[i][8] = float(list1_avg[i][8]) + float(list1[i+rep*(rowsPerRepetition)][8])   #networkingTime
      #2nd list
      list2_avg[i][1] = float(list2_avg[i][1]) + float(list2[i+rep*(rowsPerRepetition)][1])   #singleLayerInfTime
      list2_avg[i][6] = float(list2_avg[i][5]) + float(list2[i+rep*(rowsPerRepetition)][6])   #singleLayerInfTimeProf
  #AVG on reptetitions
  for i in range(1, rowsPerRepetition): 
    list1_avg[i][1] = str(list1_avg[i][1]/repetitions)   #1stInfTime
    list1_avg[i][2] = str(list1_avg[i][2]/repetitions)   #2ndInfTime
    list1_avg[i][3] = str(list1_avg[i][3]/repetitions)   #oscarJobTime
    list1_avg[i][4] = str(list1_avg[i][4]/repetitions)   #kubePodTime
    list1_avg[i][5] = str(list1_avg[i][5]/repetitions)   #tensorSaveTime
    list1_avg[i][6] = str(list1_avg[i][6]/repetitions)   #tensorLoadTime
    list1_avg[i][8] = str(list1_avg[i][8]/repetitions)   #networkingTime
    #2nd list
    list2_avg[i][1] = str(list2_avg[i][1]/repetitions)   #singleLayerInfTime
    list2_avg[i][6] = str(list2_avg[i][6]/repetitions)   #singleLayerInfTimeProf

  #Calculate ERROR on profiling data (InferenceError)
  listInfError = [0]*(rowsPerRepetition+1)
  for i in range(3, rowsPerRepetition): #starts from 3, because the first line is the header, while the next two are NO_SPLIT rows
    #for each layer(row), calculate the sum of singleLayerInfTimeProf's until the current considered layer
    sumLayerInfTimes = 0
    for j in range(3, i+1): 
      #sumLayerInfTimes = sumLayerInfTimes + float(list2_avg[j][5])    #error calculated on singleLayerInfTimeProf
      sumLayerInfTimes = sumLayerInfTimes + float(list2_avg[j][1])    #error calculated on singleLayerInfTime
    
    #Now calculate the error compared to 1stInfTime
    listInfError[i] = (sumLayerInfTimes - float(list1_avg[i][1])) / float(list1_avg[i][1])
  
  #Now get the Average InferenceError
  avgInfError = 0
  avgAbsInfError = 0
  for i in range(3, rowsPerRepetition): 
    avgInfError = avgInfError + listInfError[i]
    avgAbsInfError = avgAbsInfError + np.abs(listInfError[i])
  avgInfError = avgInfError / (rowsPerRepetition-2)
  avgAbsInfError = avgAbsInfError / (rowsPerRepetition-2)

  #Unite the two tables into a third cvs file
  import math
  with open(FINAL_RESULTS_CSV_FILE, 'w', newline='') as csvfile3:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'Memory', 'MACs', 'SingleLayerSum-Splitted', "singleLayerInfTimeProf"]
    cvswriter = csv.DictWriter(csvfile3, fieldnames=fieldnames)
    cvswriter.writeheader()

    for i in range(1,len(list1)): 
      if i % rowsPerRepetition == 0:
        cvswriter.writeheader()
      else:
        cvswriter.writerow({"SplitLayer":list1[i][0], "1stInfTime":list1[i][1], "2ndInfTime":list1[i][2], "oscarJobTime":list1[i][3], "kubePodTime":list1[i][4],
                          "tensorSaveTime":list1[i][5], "tensorLoadTime":list1[i][6], "tensorLength":list1[i][7], "networkingTime":list1[i][8], 
                          "singleLayerInfTime":list2[i][1], "OpType":list2[i][2], "NrParameters":list2[i][3], "Memory":list2[i][4], "MACs":list2[i][5],
                          "SingleLayerSum-Splitted": str(singleLayerTimeSum[math.floor(i/rowsPerRepetition)] - (float(list1[i][1]) + float(list1[i][2]))),
                          "singleLayerInfTimeProf": list2[i][6]})

  #Unite the two tables into a fourth cvs file, averaging the time measurements
  import math
  with open(AVG_RESULTS_CSV_FILE, 'w', newline='') as csvfile3:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'Memory', 'MACs', 'singleLayerInfTimeProf',
                  'InferenceError', 'AbsInferenceError', 'AvgError', 'AvgAbsError']
    cvswriter = csv.DictWriter(csvfile3, fieldnames=fieldnames)
    cvswriter.writeheader()

    for i in range(1, rowsPerRepetition): 
      cvswriter.writerow({"SplitLayer":list1[i][0], "1stInfTime":list1_avg[i][1], "2ndInfTime":list1_avg[i][2], "oscarJobTime":list1_avg[i][3], "kubePodTime":list1_avg[i][4],
                          "tensorSaveTime":list1_avg[i][5], "tensorLoadTime":list1_avg[i][6], "tensorLength":list1[i][7], "networkingTime":list1_avg[i][8], 
                          "singleLayerInfTime":list2_avg[i][1], "OpType":list2[i][2], "NrParameters":list2[i][3], "Memory":list2[i][4], "MACs":list2[i][5],
                          "singleLayerInfTimeProf": list2_avg[i][6], "InferenceError": str(listInfError[i]), "AbsInferenceError": str(np.abs(listInfError[i])), 
                          "AvgError": str(avgInfError) if i==1 else '', "AvgAbsError": str(avgAbsInfError) if i==1 else ''})

  #TESTING
  #print("Plotting the results..")
  #plot_results(AVG_RESULTS_CSV_FILE)

def onnx_run_profiler(onnx_file, onnx_path, image_file, image_batch, img_size_x, img_size_y, is_grayscale, 
                          repetitions, exec_provider, device_type, server_url, xml_file = None):
  '''
  Run with the (onnxruntime)profiling function the full onnx model on both Edge and Cloud and profile the execution times layer by layer as well as
  all the other data such as FLOPS, Nr. of Operations ecc..

  :param onnx_file: the full unsplitted ONNX file (used to gather usefull information)
  :param onnx_path: the path to the collection of models were to find the correct one to use for the inference
  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :param repetition: specifies the number of repetitions to execute
  :param exec_provider: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  '''
  #Default Argument Values
  if is_grayscale == None: is_grayscale = False
  if repetitions == None: repetitions = 1

  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  # Process input data (image or batch of images)
  inputData = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale, model_onnx.graph.input[0])
  batchSize = inputData.shape[0]

  # Run the model on Edge
  # Process and get the ProfiligTable by running at inference the full model (A WARMUP execution is also performed before)
  print("Warmup inference (Edge)..")
  resData, _ = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=True, xml_file=xml_file)
  print("Run the inference of the whole layer locally (Edge)..")
  resDataEdge, profilingTableEdge = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=True, xml_file=xml_file)
  print("Finished inference of the whole layer locally (Edge)!")
  profilingTableCloud = None

  #Execute at inference the whole model on the Cloud
  try:
    (t_1st_inf, t_2nd_inf, t_oscar_job, t_kube_pod, tensor_lenght, 
     t_tensor_save, t_tensor_load, t_networking, profilingTableCloud) = onnx_run_complete(onnx_file,  #it should be onnx_path, but since we skip the local execution and don't use splits, we pass the full model
                                                                                        "PROFILING", 
                                                                                        image_file, 
                                                                                        image_batch, 
                                                                                        img_size_x, 
                                                                                        img_size_y, 
                                                                                        is_grayscale,
                                                                                        exec_provider,
                                                                                        device_type,
                                                                                        server_url,
                                                                                        xml_file=None) 
  except Exception as e:
    print("Error on executin RUN Complete(Profiling) cycle: " + str(e))  
  finally:
    print("Finished inference (with Profiling) of the whole layer on the Cloud (OSCAR)..")

  #Get the list of all the layers we need (the ones that are used for making the splits)
  splitNodes = []
  splitNodesCompatibility = []    #for OLD onnxruntime versions, we need the names of the nodes to be saved differently (attribute name instead of output from the onnx graph)
  for layer, lnode in enumerate_model_node_outputs(model_onnx, add_node = True):
    #Ignore the first and the last layer
    if layer != list(enumerate_model_node_outputs(model_onnx))[0] and layer != list(enumerate_model_node_outputs(model_onnx))[-1]:
      splitLayer = layer.replace("/", '-').replace(":", '_')
      print("Search for: " + splitLayer)
      for dir in os.listdir(onnx_path):
        if dir.find("_on_") > 0:
          index = dir.index('_on_')
          d = dir[index+4:]
          #print("Check: " + d)
          if d == splitLayer:
            print("Found Layer: " + d)
            splitNodes.append(layer)
            splitNodesCompatibility.append(lnode.name)

  # Get the Inference Time of each layer (it can be also a sequence of nodes) by analyzing the profiling Table - Edge & Cloud
  if version.parse(onnxruntime.__version__) < version.parse("1.10.0"):
    #for old onnxruntime versions, we need different names
    dictSingleLayerInfProfilingEdge = getSingleLayerExecutionTimeTable(model_onnx, splitNodesCompatibility, profilingTableEdge)
    #update keys with the node names that we need
    for i in range(len(splitNodes)):
      dictSingleLayerInfProfilingEdge[splitNodes[i]] = dictSingleLayerInfProfilingEdge[splitNodesCompatibility[i]]
      del dictSingleLayerInfProfilingEdge[splitNodesCompatibility[i]]
  else:
    dictSingleLayerInfProfilingEdge = getSingleLayerExecutionTimeTable(model_onnx, splitNodes, profilingTableEdge)
  dictSingleLayerInfProfilingCloud = getSingleLayerExecutionTimeTable(model_onnx, splitNodes, profilingTableCloud)
      
  #Iterate through inputs of the graph and get operation types
  dictOperations = {}
  for node in model_onnx.graph.node:
    name = node.output[0]
    op_type = node.op_type
    if name in splitNodes:
      # Get OperationType
      dictOperations[name] = op_type 

  #get output tensor at each layer:
  #https://github.com/microsoft/onnxruntime/issues/1455

  # add all intermediate outputs to onnx net
  ort_session = onnxruntime.InferenceSession(onnx_file)
  org_outputs = [x.name for x in ort_session.get_outputs()]

  model = onnx.load(onnx_file)
  for node in model.graph.node:
      for output in node.output:
          if output not in org_outputs:
              model.graph.output.extend([onnx.ValueInfoProto(name=output)])

  # excute onnx
  ort_session = onnxruntime.InferenceSession(model.SerializeToString())
  outputs = [x.name for x in ort_session.get_outputs()]
  #in_img = np.fromfile('<you path>/input_img.raw', dtype=np.float32).reshape(1,3,511,511)
  inputs = onnx_get_true_inputs(model_onnx)
  ort_outs = ort_session.run(outputs, {str(inputs[0]): inputData} )        
  from collections import OrderedDict
  ort_outs = OrderedDict(zip(outputs, ort_outs))

  dictTensorLength = {}
  dictNrFilters = {}
  for layer in splitNodes:
    shape = ort_outs[layer].shape
    if len(shape) == 4:
      dictTensorLength[layer] = shape[0]*shape[1]*shape[2]*shape[3]
    else:
      dictTensorLength[layer] = shape[0]*shape[1]
    dictNrFilters[layer] = shape[1]

    
  #Now we split the layer in single blocks and run them individually in order to get the inference time per layer
  onnx_model_split_all_singlenode(onnx_file, splitNodes)     #TODO: we can also get all the info without splitting perhaps

  #Run at inference all the single layer models generated and get the inference execution time and the Nr. of Parameters
  #Acquire Nr. of Parameters and FLOPS per each layer in the
  dictNrParams = {}
  dictFLOPS = {}
  for layer in splitNodes:
    #Ignore the first layer since it will be the input
    if True: #layer != list(dictTensors.keys())[0]:
      file = "SingleLayerSplits/"+layer.replace("/", '-').replace(":", '_')+'.onnx'

      if os.path.exists(file):
        #Get the Nr. of Parameters
        single_layer_model_onnx = load_onnx_model(file)
        params = calculate_params(single_layer_model_onnx)
        dictNrParams[layer] = params
        #print('Number of params:', params)

        #For every model calc FLOPS
        '''onnx_json = convert(
          onnx_graph=single_layer_model_onnx.graph,
        )'''
        onnx_json = convert(
          input_onnx_file_path=file,
          output_json_path="test.json",
          json_indent=2,
        )
        dictNodeFLOPS = calc_flops(onnx_json, batchSize)

        #Sum all the FLOPS of the nodes inside the Single Layer Model
        dictFLOPS[layer] = 0
        for node in dictNodeFLOPS:
          dictFLOPS[layer] = dictFLOPS[layer] + dictNodeFLOPS[node]
        print(file + " FLOPS: " + str(dictFLOPS[layer]))

  #Open an cvs file to save the results
  with open(PROFILER_RESULTS_CSV_FILE, 'w', newline='') as csvfile:
    fieldnames = ['SplitLayer', 'TensorLength', 'OpType', 'NrParameters', 'NrFilters', 'FLOPS', "LayerInfTimeEdge", "LayerInfTimeCloud"]
    cvswriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    cvswriter.writeheader()

    for layer in splitNodes: 
      cvswriter.writerow({"SplitLayer":layer.replace("/", '-').replace(":", '_'), "TensorLength":dictTensorLength[layer], "OpType":dictOperations[layer], 
                          "NrParameters":dictNrParams[layer], "NrFilters":dictNrFilters[layer], "FLOPS":dictFLOPS[layer],
                          "LayerInfTimeEdge": dictSingleLayerInfProfilingEdge[layer], "LayerInfTimeCloud": dictSingleLayerInfProfilingCloud[layer]})

def getSingleLayerExecutionTimeTable(model_onnx, splitNodes, profilingTable):
  '''
  Get a Dictionary with the SingleLayersInference Time values (for each layer we have splitted the main model) based on the 
  data we got from onnxruntime Profiling of the whole onnx model.
  
  :param model_onnx: the imported full onnx model
  :param splitNodes: the list of all the layers we have splitted the main model
  :param profilingTable: the Profiling Table
  :returns: a dictionary with the split layers names as keys and inference time calculated as value
  '''
  dictSingleLayerInfProfiling = {}
  dictTensorLengths = {}
  prevNode = ""

  #Iterate through the nodes where we have splitted the model
  for node in splitNodes:
    closestNode = getClosestNodeInProfilingTable(model_onnx, profilingTable, node)
    if closestNode != "":
      print(closestNode)
      #Get SingleLayerInference time
      dictSingleLayerInfProfiling[node] = getInfTimeBetweenNodes(profilingTable, prevNode, closestNode)/1000000

      prevNode = closestNode
    else:
      dictSingleLayerInfProfiling[node] = 0
  
  return dictSingleLayerInfProfiling

def getNextNode(model_onnx, currentNode):
  '''
  Get the name of the node that comes immediatly after the one passed as function argument.
  
  :param model_onnx: the imported full onnx model
  :param currentNode: the node we consider from the model
  :returns: the name next node
  '''
  prevNode = ""
  for n in model_onnx.graph.node:
    node = n.output[0]
    if prevNode == currentNode:
      return node

    prevNode = node
  return ""

def isNodeInProfilingTable(profilingTable, node):
  '''
  Get the closest Node in the Profiling Table to the node specified (which is one of the nodes used for slitting the model)
  
  :param profilingTable: the Profiling Table
  :param nodes: a node of the model
  :returns: True if the node is present, False otherwise
  '''
  for i in range(0, len(profilingTable)):
      #if reatched the destination node
      if node in profilingTable[i]["name"]:
        return True
      # for older versions of onnxruntime
      elif ":0" in node:
          if node[:-2] in profilingTable[i]["name"]:
            return True
  return False

def getClosestNodeInProfilingTable(model_onnx, profilingTable, node):
  '''
  Get the closest Node in the Profiling Table to the node specified (which is one of the nodes used for slitting the model).
  This research will only search two nodes deep!
  
  :param model_onnx: the imported full onnx model
  :param profilingTable: the Profiling Table
  :param nodes: one of the nodes used for slitting the model
  :returns: the name of the closest node in the ProfilingTable
  '''
  #is the node already present in the profiling table?
  if isNodeInProfilingTable(profilingTable, node):
    return node
  else:
    #try the next node in the model
    nextNode = getNextNode(model_onnx, node)
    if nextNode != "":
      if isNodeInProfilingTable(profilingTable, nextNode):
        return nextNode
      else:
        #try the next next node in the model
        nextNextNode = getNextNode(model_onnx, nextNode)
        if nextNextNode != "":
          if isNodeInProfilingTable(profilingTable, nextNextNode):
            return nextNextNode
  return ""

def getInfTimeBetweenNodes(profilingTable, startNode, endNode):
  '''
  Get the sum of inference time values from a starting node to an ending node (excluded) from the profiling table.
  If the starting node is the empty string, then start from the beging with the first node in the ProfilingTable.
  
  :param profilingTable: the Profiling Table
  :param startNode: the staring node considered
  :param endNode: the last node considered (excluded)
  :returns: the inference time value in micro seconds
  '''
  infTime = 0

  #from the begining
  if startNode == "":
    for i in range(2, len(profilingTable)):     #ignore the first two, beacuse they are session instances instead of code node execution
      #if reatched the destination node
      if endNode in profilingTable[i]["name"]:
        return infTime
      else:
        infTime = infTime + profilingTable[i]["dur"]
  else:
    startIndex = 0
    #find starting index
    for i in range(0, len(profilingTable)):
      #if reatched the destination node
      if startNode in profilingTable[i]["name"]:
        startIndex = i
        break

    for i in range(startIndex, len(profilingTable)):
      #if reatched the destination node
      if endNode in profilingTable[i]["name"]:
        return infTime
      else:
        infTime = infTime + profilingTable[i]["dur"]

  return 0

if __name__ == "__main__":
  main()
