from numpy.core.numeric import True_
import numpy as np
import json
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sqlalchemy import true
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import shutil

RESULTS_CSV_FILE = 'time_table.csv'
RESULTS_CSV_FILE2 = 'time_table_2.csv'
FINAL_RESULTS_CSV_FILE = 'time_table_final.csv'
AVG_RESULTS_CSV_FILE = 'time_table_avg.csv'
PROFILER_RESULTS_CSV_FILE = 'profiler_time_table.csv'
INPUT_PICKLE_FILE = 'input.txt'
OUTPUT_PICKLE_FILE = 'results.txt'
NEUTRON_INSTALLATION_PATH = '/snap/bin/neutron'

MODEL_SPLIT_FIRST_FILE = 'first_half.onnx'
MODEL_SPLIT_SECOND_FILE = 'second_half.onnx'

def onnx_import_data(image_file, image_batch, img_size_x, img_size_y, is_grayscale = False):
  '''
  Import Data (Images) - Imports the Image or Image Batch, turns it into an np array and saves ot
  on a pickle file that can be later used as input by onnx_first_inference.py

  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  '''
  array = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale)
  data = {
    "inputData": array,
  }
  
  # Save the array on a pickle file
  with open(INPUT_PICKLE_FILE, 'wb') as f:
    pickle.dump(data, f)

def data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale = False, input_tensor = None):
  '''
  Input Data Proproccessing - Imports the Image or Image Batch and turns it into an np array

  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :returns: the np array of the image or batch of images
  '''
  #Get the input's tensor shape first
  if input_tensor != None:
    input_tensor_shape = [list(input_tensor.type.tensor_type.shape.dim)[0].dim_value,
                          list(input_tensor.type.tensor_type.shape.dim)[1].dim_value,
                          list(input_tensor.type.tensor_type.shape.dim)[2].dim_value,
                          list(input_tensor.type.tensor_type.shape.dim)[3].dim_value]
  else:
    input_tensor_shape = [1,img_size_x,img_size_y,3]
    #input_tensor_shape = [1,3,img_size_x,img_size_y]    #test force another tensor shape
    #input_tensor_shape = [1,3,img_size_y,img_size_x]    #test force another tensor shape

  # Import the single Image
  if image_file != None:
    # Load an image from file
    image = load_img(image_file, target_size=(img_size_x, img_size_y), grayscale=is_grayscale)

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    #Reshape the Image based on the input's tensor shape
    if input_tensor_shape[3] == 3:
      print("Image of shape: (1,x,y,3)")

      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    elif input_tensor_shape[1] == 3:
      print("Image of shape: (1,3,y,x)")

      # reshape data for the model
      image = image.reshape((1, image.shape[2], image.shape[1], image.shape[0]))
    else:
      print("Default Image shape: (1,x,y,3)")

      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the model
    #image = preprocess_input(image)
    input = np.array(image).astype(np.float32)  # Note the extra brackets to create 1x10
    input_img = np.array(image).astype(np.float32)  # Note the extra brackets to create 1x10
    print(image.shape) 
    return input_img   
  # Import the Batch of Images
  elif image_batch != None:
    i = 0
    list_img = []
    for img_name in os.listdir(image_batch):
      #print(img_name)
      new_img = load_img(image_batch+'/'+img_name, target_size=(img_size_x, img_size_y), grayscale=is_grayscale)
      new_img = img_to_array(new_img)
      #new_img = new_img.reshape((1, new_img.shape[0], new_img.shape[1], new_img.shape[2]))

      #Reshape the Image based on the input's tensor shape
      if input_tensor_shape[3] == 3:
        #Image of shape: (1,x,y,3)
        new_img = new_img.reshape((1, new_img.shape[0], new_img.shape[1], new_img.shape[2]))
      elif input_tensor_shape[1] == 3:
        #Image of shape: (1,3,y,x)
        new_img = new_img.reshape((1, new_img.shape[2], new_img.shape[1], new_img.shape[0]))
      else:
        #Default image shape: (1,x,y,3)
        new_img = new_img.reshape((1, new_img.shape[0], new_img.shape[1], new_img.shape[2]))
     
      list_img.append(new_img)
      i = i + 1

    tuple_img = tuple(list_img)
    batch_img = np.vstack(tuple_img)
    print(batch_img.shape)
    return batch_img
  else:
    return None

def plot_results(results_file):
  '''
  Plots the results of the Inference Cycle that are saved on a CSV file (only the first cycle if there are multiple repetitions)
  ResultsFile - FieldNames:'SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLenght'

  :param results_file: the path to the CSV file where the results are saved
  '''
  N = 0
  xTicks = []
  data_inf1_oscar_job = []
  data_inf1_inf2 = []
  data_inf1_kube_pod = []
  with open(results_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',') 
    for row in csv_reader:
      if not N == 0:
        # Discard repetitions
        if row[0] == 'SplitLayer':  
          break
        #print(f'\t{row[0]},  {row[1]}, {row[2]}, {row[3]}, {row[4]}.')
        xTicks.append(row[0])
        t_1st_inf = float(row[1])
        t_2nd_inf = float(row[2])
        t_oscar_job = float(row[3])
        t_kube_pod = float(row[4])
        t_tensor_save = float(row[5])
        t_tensor_load = float(row[6])
        tensor_lenght = float(row[7])
        t_networking = float(row[6])
        data_inf1_oscar_job.append([t_1st_inf, t_networking, t_oscar_job])
        data_inf1_inf2.append([t_1st_inf, t_tensor_save, t_networking, t_2nd_inf, t_tensor_load])
        data_inf1_kube_pod.append([t_1st_inf, t_networking, t_kube_pod])
      N += 1

  print(data_inf1_oscar_job)
  # t_1st_inf = 1st Inference Execution Time
  # t_2nd_inf = 2nd Inference Execution Time
  # t_oscar_job = 2nd Inf. OSCAR JOB Exec. Time
  # t_kube_pod = 2nd Inf. Kubernets POD Exec. Time
  # tensor_lenght = 1st Inf. Tensor Lenght:
  # t_tensor_save = 1st Inf. Tensor Save Time
  # t_tensor_load = 2nd Inf. Tensor Load Time

  print("Plot the first graph where we consider also the cluster execution time..")
  # Dummy dataframe
  df = pd.DataFrame(data=data_inf1_oscar_job, columns=['1st Exec Time', 'Networking Time', '2nd Exec Time(OSCAR JOB)'])

  # Plot a stacked barchart
  ax = df.plot.bar(stacked=True)

  # Place the legend
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  plt.xticks(ticks=range(0,N-1), labels=xTicks, rotation=90)
  #plt.ylim(0, 100)
  plt.title('Execution time by layer divided between Edge and Cloud (considering cluster execution time)')
  plt.xlabel('Layer')
  plt.ylabel('Time (sec)')
  plt.show()
  #plt.figure().savefig("Figure_1.png")

  print("Plot the second graph where we consider also the kubernetes pod execution time..")
  # Dummy dataframe
  df = pd.DataFrame(data=data_inf1_kube_pod, columns=['1st Exec Time', 'Networking Time', '2nd Exec Time(Kubernetes POD)'])

  # Plot a stacked barchart
  ax = df.plot.bar(stacked=True)

  # Place the legend
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  plt.xticks(ticks=range(0,N-1), labels=xTicks, rotation=90)
  #plt.ylim(0, 100)
  plt.title('Execution time by layer divided between Edge and Cloud (considering pod execution time)')
  plt.xlabel('Layer')
  plt.ylabel('Time (sec)')
  plt.show()
  #plt.figure().savefig("Figure_2.png")

  print("Plot the third graph where we don't consider the cluster execution time..")
  # Dummy dataframe
  df = pd.DataFrame(data=data_inf1_inf2, columns=['1st Inf Exec Time', 'TensorSave Time', 'Networking Time', '2nd Exec Time', 'TensorLoad Time'])

  # Plot a stacked barchart
  ax = df.plot.bar(stacked=True)

  # Place the legend
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  plt.xticks(ticks=range(0,N-1), labels=xTicks, rotation=90)
  #plt.ylim(0, 100)
  plt.title('Execution time by layer divided between Edge and Cloud')
  plt.xlabel('Layer')
  plt.ylabel('Time (sec)')
  plt.show()
  #plt.figure().savefig("Figure_3.png")

def calc_flops(onnx_json, batchSize):
  '''
  Calculate the FLOPS of a given onnx model. It expects in input the JSON version of the onnx's graph.

  :param onnx_json: the JSON version of the onnx's graph
  :returns: a dictionay with the flops for every node in the onnx model
  '''
  dictNodeFLOPS = {}
  #Iterate all the nodes of the Single Layer Model
  for node in onnx_json['graph']['node']:
    valid_node = True
    if 'input' in node.keys() and 'output' in node.keys():
      node_inputs = node['input']     #there might be more than one input
      node_output = node['output'][0]
    else:
      valid_node = False #skip this node as it's invalid for flops calculation (usually a constant or similar)
    node_op_type = node['opType']
    if 'name' in node.keys():
      node_name = node['name']
    else:
      node_name = node_output #in some models, nodes don't have a name..

    if not valid_node:
      dictNodeFLOPS[node_name] = 0
      continue

    #Calculate FLOPS differently based on the OperationType
    if (node_op_type == "Clip" or 
        node_op_type == "Relu" or 
        node_op_type == "LeakyRelu" or
        node_op_type == "Sigmoid" or
        node_op_type == "Tanh" or
        node_op_type == "BatchNormalization"):
      #FLOPS = 3+Cout (for forward pass)

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
            Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])

            #Calc FLOPS
            dictNodeFLOPS[node_name] = 3*Cout
            break
    elif node_op_type == "Conv":
      #FLOPS = Hf*Wf*Cin*Cout (for forward pass)
      Hf,Wf,Cin,Cout = 1,1,1,1#default

      #Get KernelShape (here we can also get the pads, strides, ecc)
      for attr in node['attribute']:
        if attr['name'] == 'kernel_shape':
          Hf = int(attr['ints'][0])
          Wf = int(attr['ints'][1])
          break

      #Get Cin,Hin,Win dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Check for each node input if it corresponds with a valueInfo instance
        #That way we are basically searching for the output info of the node that is at the input of the current node
        for input in node_inputs:
          if info['name'] == input:
            if 'shape' in info['type']['tensorType']:
              Cin = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
              break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
            Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hf*Wf*Cin*Cout
    elif (node_op_type == "MaxPool" or 
          node_op_type == "LpPool" or 
          node_op_type == "AveragePool" or
          node_op_type == "GlobalMaxPool" or 
          node_op_type == "GlobalAveragePool"):
      #FLOPS = Hf*Wf*Cout (for forward pass)
      Hf,Wf,Cout = 1,1,1#default

      #Get KernelShape (here we can also get the pads, strides, ecc)
      if 'attribute' in node:
        for attr in node['attribute']:
          if attr['name'] == 'kernel_shape':
            Hf = int(attr['ints'][0])
            Wf = int(attr['ints'][1])
            break
      else:
        #No attribute in node, we get the kernel dimentions from the previous node
        for info in onnx_json['graph']['valueInfo']:
          #Check for each node input if it corresponds with a valueInfo instance
          #That way we are basically searching for the output info of the node that is at the input of the current node
          for input in node_inputs:
            if info['name'] == input:
              if 'shape' in info['type']['tensorType']:
                if len(info['type']['tensorType']['shape']['dim']) == 4:
                  Hf = int(info['type']['tensorType']['shape']['dim'][2]['dimValue']) #only in this case Hf=Hin
                  Wf = int(info['type']['tensorType']['shape']['dim'][3]['dimValue']) #only in this case Wf=Win
                break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
            Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hf*Wf*Cout
    elif (node_op_type == "BatchNormalization" or 
          node_op_type == "LpNormalization"):
      #FLOPS = 5*Cout + Cn - 2 (for forward pass)
      Cout, Cn = 1,1  #default

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
            Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            if 'dimValue' in info['type']['tensorType']['shape']['dim'][0]:
              Cn = info['type']['tensorType']['shape']['dim'][0]['dimValue']    #TODO: try dimValue first and if it's not working then dimParam
            else:
              Cn = info['type']['tensorType']['shape']['dim'][0]['dimParam']

            if Cn.startswith("unk_"):
              Cn = batchSize    #Use the dimension of the batch used to run the RUN ALL comand instead of 1
            else:
              Cn = int(Cn)
            break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = 5*Cout + Cn - 2
    elif (node_op_type == "SoftmaxCrossEntropyLoss" or 
          node_op_type == "NegativeLogLikelihoodLoss"):
      #FLOPS = 4*Cout - 1 (for forward pass)
      Cout, Cn = 1,1,1  #default

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
            Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = 4*Cout - 1
    #MatMul
    elif (node_op_type == "MatMul"): #or node_op_type == "FC"
      #FLOPS = Hin*Win*Cin*Cout (for forward pass)
      Hin,Win,Cin,Cout = 1,1,1,1#default

      #Get Cin,Hin,Win dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Check for each node input if it corresponds with a valueInfo instance
        #That way we are basically searching for the output info of the node that is at the input of the current node
        for input in node_inputs:
          if info['name'] == input:
            if 'shape' in info['type']['tensorType']:
              Cin = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
              if len(info['type']['tensorType']['shape']['dim']) == 4:
                Hin = int(info['type']['tensorType']['shape']['dim'][2]['dimValue'])
                Win = int(info['type']['tensorType']['shape']['dim'][3]['dimValue'])
              break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
            Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hin*Win*Cin*Cout
    #Add          
    elif (node_op_type == "Add" or 
          node_op_type == "Mul" or
          node_op_type == "Div" or
          node_op_type == "Sub"):
      #FLOPS = Hout*Wout*Cout (for forward pass)
      Hout,Wout,Cout = 1,1,1#default

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
            Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            if len(info['type']['tensorType']['shape']['dim']) == 4:
              Hout = int(info['type']['tensorType']['shape']['dim'][2]['dimValue'])
              Wout = int(info['type']['tensorType']['shape']['dim'][3]['dimValue'])
            break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hout*Wout*Cout
    else:
      print("WARNING! This type of Opeartion hasn't been recognized by the FLOPS calcultion algorithm! Please add support for: " + node_op_type)
      jsonFile = open(node_op_type+".json", "w")
      jsonString = json.dumps(onnx_json)
      jsonFile.write(jsonString)
      jsonFile.close()

  return dictNodeFLOPS

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


def isLayerInFilterList(layer, filter_layers):
  for flayer in filter_layers:
    if layer == flayer:
      return True
  return False

def prepare_ml_dataset(input_csv_file, output_csv_file, filter_layers, filter_nr_nodes):
  ignoreLastNRowsPerRep = 3
  if filter_nr_nodes == None:
    filter_nr_nodes = 1000000
  else:
    filter_nr_nodes = int(filter_nr_nodes)

  #Get the data from the first two cvs files
  listRows = []
  with open(input_csv_file, 'r', newline='') as csvfile1:
    reader = csv.reader(csvfile1, delimiter=",")
    for i, line in enumerate(reader):
      listRows.append(line)

  #Create the output CSV File
  with open(output_csv_file, 'w', newline='') as csvfile2:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'NrNodes', 'Memory', 'MACs', 'SingleLayerSum-Splitted', "singleLayerInfTimeProf"]
    cvswriter = csv.DictWriter(csvfile2, fieldnames=fieldnames)
    cvswriter.writeheader()

    for i in range(1,len(listRows)): 
      '''if (listRows[i][0] != "SplitLayer" and
          listRows[i][0] != "NO_SPLIT" and
          listRows[i][0] != "sequential-mobilenetv2_1.00_160-Conv_1_bn-FusedBatchNormV3_0" and
          listRows[i][0] != "sequential-global_average_pooling2d-Mean_0" and
          listRows[i][0] != "sequential-dense-MatMul_0" and
          listRows[i][0] != "resnet50-avg_pool-Mean_0" and
          listRows[i][0] != "resnet50-predictions-MatMul_0" and
          listRows[i][0] != "resnet50-predictions-BiasAdd_0" and
          listRows[i][0] != "resnet50-avg_pool-Mean_Squeeze__614_0" and
          listRows[i][0] != "resnet101-avg_pool-Mean_0" and
          listRows[i][0] != "resnet101-avg_pool-Mean_Squeeze__1224_0" and
          listRows[i][0] != "resnet101-predictions-MatMul_0" and
          listRows[i][0] != "resnet101-predictions-BiasAdd_0" and
          listRows[i][0] != "densenet121-avg_pool-Mean_0" and
          listRows[i][0] != "densenet121-avg_pool-Mean_Squeeze__1220_0" and
          listRows[i][0] != "densenet121-predictions-MatMul_0" and
          listRows[i][0] != "densenet121-predictions-BiasAdd_0" and
          listRows[i][0] != "sequential-dense_1-MatMul_0" and   ##vgg16
          listRows[i][0] != "sequential-dense_1-BiasAdd_0" and
          listRows[i][0] != "sequential-dense_2-MatMul_0" and
          listRows[i][0] != "sequential-dense_2-BiasAdd_0" and
          #filter non CONV layers - Temporary
          listRows[i][0] != "resnet101-conv1_bn-FusedBatchNormV3_0" and
          listRows[i][0] != "resnet101-pool1_pad-Pad_0" and
          listRows[i][0] != "densenet121-conv1-bn-FusedBatchNormV3_0" and
          listRows[i][0] != "densenet121-zero_padding2d_1-Pad_0" and
          listRows[i][0] != "densenet121-pool4_conv-Conv2D_0" and
          listRows[i][0] != "densenet121-pool3_conv-Conv2D_0" and
          listRows[i][0] != "densenet121-pool2_conv-Conv2D_0" and
          listRows[i][0] != "sequential-conv2d_12-BiasAdd_0" and
          listRows[i][0] != "sequential-conv2d_6-BiasAdd_0" and
          listRows[i][0] != "sequential-conv2d_1-BiasAdd_0" and
          listRows[i][0] != "sequential-conv2d_9-BiasAdd_0" and
          listRows[i][0] != "sequential-conv2d_3-BiasAdd_0" and
          listRows[i][0] != "sequential-dense-BiasAdd_0" and
          listRows[i][0] != "sequential-dense-MatMul_0" and
          listRows[i][0] != "sequential-max_pooling2d_4-MaxPool__76_0" and
          listRows[i][0] != "sequential-max_pooling2d_4-MaxPool_0" and
          listRows[i][0] != "sequential-flatten-Reshape_0" and
          listRows[i][0] != "input.4" and
          listRows[i][0] != "input.12" and
          listRows[i][0] != "input.20" and
          listRows[i][0] != "input.28" and
          listRows[i][0] != "input.36" and
          listRows[i][0] != "input.44" and
          listRows[i][0] != "input.52" and
          listRows[i][0] != "input.60" and
          listRows[i][0] != "input.68" and
          listRows[i][0] != "input.76" and
          listRows[i][0] != "input.84" and
          listRows[i][0] != "input.92" and
          listRows[i][0] != "input.100" and
          listRows[i][0] != "input.184" 
          ):'''
      if isLayerInFilterList(listRows[i][0], filter_layers) == False and int(listRows[i][12]) <= filter_nr_nodes:
        cvswriter.writerow({"SplitLayer":listRows[i][0], "1stInfTime":listRows[i][1], "2ndInfTime":listRows[i][2], "oscarJobTime":listRows[i][3], "kubePodTime":listRows[i][4],
                          "tensorSaveTime":listRows[i][5], "tensorLoadTime":listRows[i][6], "tensorLength":listRows[i][7], "networkingTime":listRows[i][8], 
                          "singleLayerInfTime":listRows[i][9], "OpType":listRows[i][10], "NrParameters":listRows[i][11],"NrNodes":listRows[i][12], "Memory":listRows[i][13], "MACs":listRows[i][14],
                          "SingleLayerSum-Splitted": listRows[i][15], "singleLayerInfTimeProf": listRows[i][16]})


def predictionProfiling(input_csv_file, input_avg_csv_file, output_csv_file):
  ignoreFirstNRows = 15#2   ##Includes Header

  #Get the data from the  full cvs file
  listRows = []
  with open(input_csv_file, 'r', newline='') as csvfile1:
    reader = csv.reader(csvfile1, delimiter=",")
    for i, line in enumerate(reader):
      listRows.append(line)

  #Get the data from the average cvs file
  listRowsAvg = []
  with open(input_avg_csv_file, 'r', newline='') as csvfile2:
    reader = csv.reader(csvfile2, delimiter=",")
    for i, line in enumerate(reader):
      listRowsAvg.append(line)

  #Create the output CSV File
  with open(output_csv_file, 'w', newline='') as csvfile3:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'NrNodes', 'Memory', 'MACs', 'singleLayerInfTimeProf',
                  'predictedSingleLayerInfTime', 'predictionError', 'sumSingleLayerInfTimes', '1stInfTimeError']
    cvswriter = csv.DictWriter(csvfile3, fieldnames=fieldnames)
    cvswriter.writeheader()

    skippedLayers = 0
    sumSingleLayerInfTimes = 0
    for i in range(1,len(listRowsAvg)): 
      # Skip Header, NO_SPLIT Layers and the last fully connected layers
      if (listRowsAvg[i][0] != "SplitLayer" and
          listRowsAvg[i][0] != "NO_SPLIT" and
          listRowsAvg[i][0] != "sequential-mobilenetv2_1.00_160-Conv_1_bn-FusedBatchNormV3_0" and
          listRowsAvg[i][0] != "sequential-global_average_pooling2d-Mean_0" and
          listRowsAvg[i][0] != "sequential-dense-MatMul_0"):
        singleLayerInfTime = listRowsAvg[i][9]
        firstInfTime = listRowsAvg[i][1]
        
        if skippedLayers >= ignoreFirstNRows:
          nextLayer = listRowsAvg[i]
          predictedSingleLayerInfTime = trainAndPredictNextLayer(listRows, nextLayer, skippedLayers == ignoreFirstNRows)  

          # Make a copy of the dataset used for training
          newDatasetName = "data_train_" + nextLayer[0] + ".csv"
          res = os.system("cp a-MLLibrary-master/inputs/tmp/data_train.csv a-MLLibrary-master/inputs/tmp/" + newDatasetName)

          predictionError = 1.0 - (predictedSingleLayerInfTime / float(singleLayerInfTime))      # singleLayerInfTime/1 = predictedSingleLayerInfTime/x
          sumSingleLayerInfTimes = sumSingleLayerInfTimes + predictedSingleLayerInfTime
          firstInfTimeError = 1.0 - (sumSingleLayerInfTimes / float(firstInfTime))               # firstInfTime/1 = sumSingleLayerInfTimes/x

          cvswriter.writerow({"SplitLayer":listRowsAvg[i][0], "1stInfTime":firstInfTime, "2ndInfTime":listRowsAvg[i][2], "oscarJobTime":listRowsAvg[i][3], "kubePodTime":listRowsAvg[i][4],
                              "tensorSaveTime":listRowsAvg[i][5], "tensorLoadTime":listRowsAvg[i][6], "tensorLength":listRowsAvg[i][7], "networkingTime":listRowsAvg[i][8], 
                              "singleLayerInfTime":singleLayerInfTime, "OpType":listRowsAvg[i][10], "NrParameters":listRowsAvg[i][11],"NrNodes":listRowsAvg[i][12], "Memory":listRowsAvg[i][13], 
                              "MACs":listRowsAvg[i][14], 'singleLayerInfTimeProf': listRowsAvg[i][15], 
                              "predictedSingleLayerInfTime": str(predictedSingleLayerInfTime), 
                              "predictionError": str(predictionError), 
                              "sumSingleLayerInfTimes": str(sumSingleLayerInfTimes), 
                              "1stInfTimeError": str(firstInfTimeError) 
                            })
        else:
          sumSingleLayerInfTimes = sumSingleLayerInfTimes + float(singleLayerInfTime)
        skippedLayers = skippedLayers + 1

def trainAndPredictNextLayer(listRows, nextLayer, firstTimeExecuted):
  '''
    N-1 Layers will be used for Training
    Nth Layer will be used for prediction
  '''
  #Create the output CSV File to be used as dataset for the ML Model Training (with all rows but the last one)
  with open("a-MLLibrary-master/inputs/tmp/data_train.csv", 'w', newline='') as csvfile1:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'NrNodes', 'Memory', 'MACs']
    cvswriter = csv.DictWriter(csvfile1, fieldnames=fieldnames)
    cvswriter.writeheader()

    #Find rows per repetition
    rowsPerRepetition = 0
    for i in range(1,len(listRows)): 
      if (listRows[i][0] == "SplitLayer"):
        rowsPerRepetition = i
        break

    #Find the layers that must be used for ML Model Training
    layersForTraining = []
    for i in range(1,len(listRows)): 
      #Control all the rows and discard: headers, fullyconnected layers, 
      if (listRows[i][0] != "SplitLayer" and
          listRows[i][0] != "NO_SPLIT" and
          listRows[i][0] != "sequential-mobilenetv2_1.00_160-Conv_1_bn-FusedBatchNormV3_0" and
          listRows[i][0] != "sequential-global_average_pooling2d-Mean_0" and
          listRows[i][0] != "sequential-dense-MatMul_0"):
        #Add the layers up to 'nextLayer' to the list of layers that will be used for the training
        if (listRows[i][0] != nextLayer[0] and i < rowsPerRepetition):
          layersForTraining.append(listRows[i][0])
        elif (listRows[i][0] == nextLayer[0]):
          break

    #Iterate all the rows and add the ones allowed to the dataset used for training 
    for i in range(1,len(listRows)): 
      if (listRows[i][0] in layersForTraining):
        cvswriter.writerow({"SplitLayer":listRows[i][0], "1stInfTime":listRows[i][1], "2ndInfTime":listRows[i][2], "oscarJobTime":listRows[i][3], "kubePodTime":listRows[i][4],
                          "tensorSaveTime":listRows[i][5], "tensorLoadTime":listRows[i][6], "tensorLength":listRows[i][7], "networkingTime":listRows[i][8], 
                          "singleLayerInfTime":listRows[i][9], "OpType":listRows[i][10], "NrParameters":listRows[i][11],"NrNodes":listRows[i][12], "Memory":listRows[i][13], "MACs":listRows[i][14]})

  #Create the output CSV File to be used as dataset for the ML Model for prediction
  with open("a-MLLibrary-master/inputs/tmp/data_predict.csv", 'w', newline='') as csvfile2:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLength', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'NrNodes', 'Memory', 'MACs']
    cvswriter = csv.DictWriter(csvfile2, fieldnames=fieldnames)
    cvswriter.writeheader()

    cvswriter.writerow({"SplitLayer":nextLayer[0], "1stInfTime":nextLayer[1], "2ndInfTime":nextLayer[2], "oscarJobTime":nextLayer[3], "kubePodTime":nextLayer[4],
                        "tensorSaveTime":nextLayer[5], "tensorLoadTime":nextLayer[6], "tensorLength":nextLayer[7], "networkingTime":nextLayer[8], 
                        "singleLayerInfTime":nextLayer[9], "OpType":nextLayer[10], "NrParameters":nextLayer[11],"NrNodes":nextLayer[12], "Memory":nextLayer[13], "MACs":nextLayer[14]})

  #delete a-MLLibrary-master/tmp_output directory
  try:
    #shutil.rmtree("a-MLLibrary-master/tmp_output")
    shutil.rmtree("a-MLLibrary-master/output_tmp_prediction")
  except Exception as e:
    print(e)
  
  #Training
  if firstTimeExecuted:
    try:
      shutil.rmtree("a-MLLibrary-master/tmp_output")
    except Exception as e:
      print(e)
    print("\nTraining ML Model for layer: " + nextLayer[0] + "\n") 
    #python3 a-MLLibrary-master/run.py -c a-MLLibrary-master/example_configurations/tmp/train.ini -o a-MLLibrary-master/tmp_output
    res = os.system("python3 a-MLLibrary-master/run.py -c a-MLLibrary-master/example_configurations/tmp/train.ini -o a-MLLibrary-master/tmp_output")
  else: 
    print("\nNo need to train, model already exists\n") 

  #Predict
  print("\nPredict layer: " + nextLayer[0] + "\n") 
  #python3 a-MLLibrary-master/predict.py -c a-MLLibrary-master/example_configurations/tmp/predict.ini -r a-MLLibrary-master/tmp_output/best.pickle -o a-MLLibrary-master/output_tmp_prediction/prediction.csv
  res = os.system("python3 a-MLLibrary-master/predict.py -c a-MLLibrary-master/example_configurations/tmp/predict.ini -r a-MLLibrary-master/tmp_output/best.pickle -o a-MLLibrary-master/output_tmp_prediction")

  # Get Prediction
  listPred = []
  firstInfTimePredicted = 0
  with open("a-MLLibrary-master/output_tmp_prediction/prediction.csv", 'r', newline='') as csvfile3:
    reader = csv.reader(csvfile3, delimiter=",")
    for i, line in enumerate(reader):
      listPred.append(line)

  for i in range(1,len(listPred)): 
    if listPred[i][0] != "real":
      firstInfTimePredicted = float(listPred[i][1])
      break 

  #delete a-MLLibrary-master/tmp_output directory
  try:
    #shutil.rmtree("a-MLLibrary-master/tmp_output")
    shutil.rmtree("a-MLLibrary-master/output_tmp_prediction")
  except Exception as e:
    print(e)

  return firstInfTimePredicted
