from space4aidpartitioner import SPACE4AIDPartitioner
from skl2onnx.helpers.onnx_helper import load_onnx_model
import pickle
import click

@click.command()
@click.option("--onnx_file")
@click.option("--output_path")
@click.option("--num_partitions", default=10)
@click.option("--pickle_file", default="temp/split_layers")
def main(onnx_file, output_path, num_partitions, pickle_file):
    partitionable_model = output_path
    partitioner = SPACE4AIDPartitioner(onnx_file, partitionable_model)

    #Generate a first partition of the model and get the split points
    split_layers = partitioner.get_partitions(num_partitions=num_partitions)

    #Get an ordered list of all the possible split points
    dictLayers = partitioner._get_ordered_nodes(load_onnx_model(onnx_file))
    listLayers = list(dictLayers.keys())

    #Rearrange the order of the list of the split points
    split_layers = sorted(split_layers, key=listLayers.index)

    #Save the list with the split points in a pickle file
    with open(pickle_file, "wb") as fp:   #Pickling
        pickle.dump(split_layers, fp)

if __name__ == "__main__" :
    main()
