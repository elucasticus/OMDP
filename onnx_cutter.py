from space4aidpartitioner import SPACE4AIDPartitioner
import onnx
import pickle

def main():
    onnx_file = "mobilenet.onnx"
    partitionable_model = "onnx_models/mobilenet"
    partitioner = SPACE4AIDPartitioner(onnx_file, partitionable_model)
    num_partitions = 10
    split_layers = partitioner.get_partitions(num_partitions=num_partitions)
    with open("temp/split_layers", "wb") as fp:   #Pickling
        pickle.dump(split_layers, fp)

    #with open("test", "rb") as fp:   # Unpickling
    #    split_layers = pickle.load(fp)
    
    #input_names = split_layers[3]
    #output_names = split_layers[5] 
    #print("input_names: %s" %input_names)
    #print("output_names: %s" %output_names)
    #onnx.utils.extract_model(onnx_file, partitionable_model + "/third_half.onnx", [input_names], [output_names])

if __name__ == "__main__" :
    main()
