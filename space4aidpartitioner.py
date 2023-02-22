import os
import shutil

from tqdm import tqdm

import numpy as np

from typing import List

import onnx
from onnx.utils import Extractor
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model

from google.protobuf.json_format import MessageToDict

class SPACE4AIDPartitioner():
    ''' Temporary partitioner that, given a partitionable model:
        - Generate the model partitions from the ONNX file. In particular,
            takes the partition with the lowest tensor size.
        - Generate the Python code to execute the partitions
    '''

    def __init__(self, onnx_file, partitionable_model):
        self.partitionable_model = partitionable_model
        self.onnx_file = onnx_file
    
    def get_partitions(self, num_partitions=1):
        # Load the Onnx Model
        print("\n")
        print("[AI-SPRINT]: " + "Running SPACE4AI-D-partitioner..")
        print("             " + "Maximum number of required partitions: {}".format(num_partitions))
        print("             " + "- Finding partitions of model: {}".format(self.partitionable_model))
        onnx_model = load_onnx_model(self.onnx_file)
        sorted_nodes = self._get_sorted_nodes(onnx_model=onnx_model)

        return self.onnx_model_split_first_smallest(
            sorted_nodes, onnx_model=onnx_model, number_of_partitions=num_partitions)

    def _get_node_type(self, onnx_model, node_name):
        for node in onnx_model.graph.node:
            if node_name in node.output:
                return node.op_type
    
    def _node_is_activation(self, node):
        # NOTE: add others?
        if 'relu' in node:
            return True
        if 'tanh' in node:
            return True
        if 'sigmoid' in node:
            return True
        if 'affine' in node:
            return True
        if 'elu' in node:
            return True
        if 'softsign' in node:
            return True
        if 'softplus' in node:
            return True
        return False

    def _get_sorted_nodes(self, onnx_model):
        shape_info = onnx.shape_inference.infer_shapes(onnx_model)
        shape_info_dict = {}
        output_nodes = [node.name for node in onnx_model.graph.output]
        for info in shape_info.graph.value_info:
            info_dict = MessageToDict(info)
            node_name = info_dict['name']
            # Do not consider output nodes
            if node_name in output_nodes:
                continue
            # Do not consider activation nodes TODO: to be double checked 
            node_type = self._get_node_type(
                onnx_model=onnx_model, node_name=node_name)
            if self._node_is_activation(node_type):
                continue
            # Do not consider nodes with no shape 
            # (NOTE: this is only for this implementation)
            if 'shape' in info_dict['type']['tensorType'].keys():
                shape = info_dict['type']['tensorType']['shape']
                if not shape: 
                    continue
            else:
                continue
            
            dims = [int(dim['dimValue']) for dim in shape['dim'] if 'dimValue' in dim]
            num_pixels = np.prod(dims)

            shape_info_dict[node_name] = num_pixels
        
        # Order nodes based on the output tensor
        shape_info_dict_sorted = dict(sorted(shape_info_dict.items(), key=lambda item: item[1]))

        return shape_info_dict_sorted

    def _get_true_input(self, onnx_model):
        '''
        Get the list of TRUE inputs of the ONNX model passed as argument.
        The reason for this is that sometimes "onnx.load" interprets some of the static initializers
        (such as weights and biases) as inputs, therefore showing a large list of inputs and misleading for instance
        the fuctions used for splitting.
        :param onnx_model: the already imported ONNX Model
        :returns: a list of the true inputs
        '''
        input_names = []

        initializers = [node.name for node in onnx_model.graph.initializer]

        # Iterate all inputs and check if they are valid
        for i in range(len(onnx_model.graph.input)):
            node_name = onnx_model.graph.input[i].name
            # Check if input is not an initializer, if so ignore it
            if node_name in initializers:
                continue
            else:
                input_names.append(node_name)

        return input_names

    def extract_model(
        self,
        model,
        output_path: str,
        input_names: List[str],
        output_names: List[str],
        check_model: bool = True,
    ) -> None:
        """Extracts sub-model from an ONNX model.
        The sub-model is defined by the names of the input and output tensors *exactly*.
        Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
        which is defined by the input and output tensors, should not _cut through_ the
        subgraph that is connected to the _main graph_ as attributes of these operators.
        Arguments:
            input_path (string): The path to original ONNX model.
            output_path (string): The path to save the extracted ONNX model.
            input_names (list of string): The names of the input tensors that to be extracted.
            output_names (list of string): The names of the output tensors that to be extracted.
            check_model (bool): Whether to run model checker on the extracted model.
        """
        if not output_path:
            raise ValueError("Output model path shall not be empty!")
        if not output_names:
            raise ValueError("Output tensor names shall not be empty!")

        e = Extractor(model)
        extracted = e.extract_model(input_names, output_names)

        onnx.save(extracted, output_path)
        if check_model:
            onnx.checker.check_model(output_path)

    def onnx_model_split_first_smallest(self, sorted_nodes, onnx_model=None, number_of_partitions=1):
        ''' Find all the possible partitions of the ONNX model, 
            which are stored as designs in the designs folder of the AI-SPRINT application.
        '''

        designs_folder = self.partitionable_model
        
        if onnx_model is None:
            # Load the Onnx Model
            if not os.path.exists(self.onnx_file):
                raise ValueError(f"Invalid input model path: {self.onnx_file}")
            onnx.checker.check_model(self.onnx_file)
            onnx_model = load_onnx_model(self.onnx_file)

        found_partitions = []

        #Make a split for every layer in the model
        ln = 0
        # for layer in enumerate_model_node_outputs(onnx_model):
        partitioned_layers = []
        for layer in tqdm(sorted_nodes):
            # Initialize partitions designs folders
            which_partition = '{}'.format(ln)
            partition_dir = os.path.join(designs_folder, 'split_' + which_partition + '_on_' + layer.replace("/", '-').replace(":", '_'))
            if not os.path.exists(partition_dir):
                os.makedirs(partition_dir)

            # Split and save the second half of the current partition 
            # -------------------------------------------------------
            input_names = [layer]

            output_names = []

            for i in range(len(onnx_model.graph.output)):
                output_names.append(onnx_model.graph.output[i].name)

            try:
                self.extract_model(
                    onnx_model,
                    os.path.join(partition_dir,'second_half.onnx'), 
                    input_names, output_names)
                found_partitions.append(which_partition+'_2')
            except Exception as e:
                # print(e)
                found_partitions = []
                shutil.rmtree(partition_dir)
                continue
            # -------------------------------------------------------
            
            # Split and save the first half of the current partition 
            # ------------------------------------------------------
            input_names = self._get_true_input(onnx_model) 
            
            output_names = [layer]

            self.extract_model(onnx_model, 
                               os.path.join(partition_dir, 'first_half.onnx'), 
                               input_names, output_names)

            found_partitions.append(which_partition+'_1')

            # Found first smallest partition, then break
            ln = ln + 1

            partitioned_layers.append(layer)

            if ln == number_of_partitions:
                break
        
        print("\n")        
        print("             " + "  Done! Model partitioned at layers: {}\n".format(partitioned_layers))
        return found_partitions
        # ------------------------------------------------------
        
        # ln = ln + 1