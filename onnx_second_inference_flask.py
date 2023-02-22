import onnxruntime
import onnx
import numpy as np
import argparse, sys
import pickle
import os
import time
import numpy
import json
from json import JSONEncoder


def onnx_search_and_run_second_half(onnx_models_path, onnx_model_file, data, results_file, EP_list, device = None):
  '''
  Use an input model OR search the correct one and run at inference the Second Half of the Splitted Model

  :param onnx_model_file: the ONNX file to use for the inference (only choose onnx_path or onnx_model_file)
  :param onnx_models_path: the path to the collection of models were to find the correct one to use for the inference
  :param data: the dictionary that contains the input tensor to be fed to the model
  :param results_file: specifies the file where to save the results of the inference (if not present, results will only be returned)
  :param EP_list: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  :returns: the results of the inference, the execution time of the first and second inference and the split layer used
  '''
  startTime = time.perf_counter()

  #Get the input tensor from the file
  input_tensor = data["result"]
  input_layer = data["splitLayer"].replace("/", '-').replace(":", '_')
  isNoSplitCase = (data["splitLayer"] == "NO_SPLIT")  #No split case is when we don't have to pick a model based on the split, but instead we use the whole model
  isProfilingCase = (data["splitLayer"] == "PROFILING")  #Profiling case is when we don't have to pick a model based on the split, but instead we use the whole model and activate the onnxruntime profiling
  execTime1 = data["execTime1"]
  tensorLength = data["tensorLength"]
  tensorSaveTime = data["tensorSaveTime"]
  #print(data)

  tensorLoadTime = time.perf_counter() - startTime
  #startTime = time.perf_counter() 

  #Iterate through the subdirectories to find the ONNX model splitted at our selected layer
  #print("Search for: " + input_layer)
  onnx_file = None
  if onnx_model_file == None:
    for dir in os.listdir(onnx_models_path):
      if dir.find("_on_") > 0:
        index = dir.index('_on_')
        d = dir[index+4:]
        #print("Check: " + d)
        if d == input_layer:
          #print("Found Layer: " + d)
          onnx_file = onnx_models_path + "/" + dir + "/second_half.onnx"
          break
  #Use the ONNX Model specified (don't search)
  else:
    onnx_file = onnx_model_file

  #No split case --> Get the full model instead of a splitted part
  if isNoSplitCase or isProfilingCase:
    if onnx_models_path == None:
      onnx_file = data["fullModelFile"]
    else:
      onnx_file = onnx_models_path + "/" + data["fullModelFile"]

  #Get the input and output of the model
  print("onnx_file used: " + onnx_file)
  onnx_model = onnx.load(onnx_file)
  model_input = onnx_model.graph.input[0].name 
  model_output = onnx_model.graph.output[0].name

  startTime = time.perf_counter()

  # Run the second model with the results from the first model as input (Using sclblonnx)
  '''g = so.graph_from_file(onnx_file)
  inputs = {model_input: input_tensor}
  result = so.run(g,
                  inputs=inputs,
                  outputs=[model_output]
                  )'''
  
  #Enable Profiling via ONNXRUNTIME
  so = onnxruntime.SessionOptions()
  if isProfilingCase:
    so.enable_profiling = True
  
  # Run the second model with the results from the first model as input (Using onnxruntime)
  ort_session = onnxruntime.InferenceSession(onnx_file, so, providers=EP_list, provider_options=[{'device_type' : device}])
  print('Providers:' + str(ort_session.get_providers()))
  #print('Provider Options:' + str(ort_session.get_provider_options()))
  result = ort_session.run(None, {model_input: input_tensor})  

  endTime = time.perf_counter()  

  # Profiling ends
  profilingTable = []
  if isProfilingCase:
    try:
      prof = ort_session.end_profiling()
      # and is collected in that file:
      print(prof)

      # Import the list of nodes execution logs
      with open(prof, "r") as f:
        profilingTable = json.load(f)
      #print(js[:3])

      # a tool to convert it into a table and then into a csv file
      #df = DataFrame(OnnxWholeSession.process_profiling(profilingTable))
      #df.to_csv("inference_profiling.csv", index=False)
    except Exception as e:
      print("Error while saving the profiling during first inference!" + str(e))  
      isProfilingCase = False

  
  dictData = {
    "splitLayer": input_layer,
    "execTime1": execTime1,             #1st Inference Execution Time
    "execTime2": endTime-startTime,     #2nd Inference Execution Time
    "result": result[0],
    "tensorLength": tensorLength,
    "tensorSaveTime": tensorSaveTime,
    "tensorLoadTime": tensorLoadTime
  }

  #Get the InferenceExecutionTime from the profiling if present
  if isProfilingCase:
    execTimeFromProfiling = (profilingTable[len(profilingTable)-1]['dur'] +
                              profilingTable[0]['dur'] + profilingTable[1]['dur']) / 1000000
    dictData["execTime2"] = execTimeFromProfiling
    dictData["profilingTableCloud"] = profilingTable

  if results_file != None:
    with open(results_file, 'wb') as f:
      pickle.dump(dictData, f)

  # Convert into JSON:
  # returnData = json.dumps(dictData, cls=NumpyArrayEncoder)

  # print(returnData)
  return dictData

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
