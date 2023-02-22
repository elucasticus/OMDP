from space4aidpartitioner import SPACE4AIDPartitioner

def main():
    onnx_file = "mobilenet.onnx"
    partitionable_model = "onnx_models/mobilenet"
    partitioner = SPACE4AIDPartitioner(onnx_file, partitionable_model)
    num_partitions = 10
    partitioner.get_partitions(num_partitions=num_partitions)

if __name__ == "__main__" :
    main()
