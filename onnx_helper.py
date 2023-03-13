import os, sys
import csv
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class HiddenPrints:
    """
    Suppress console output via the with keyword:\n
    with HiddenPrints():
        do something
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CsvHandler:
    """
    Manage the .csv generated by the tool
    """

    def __init__(self, csvfile):
        """
        Initialize the object of the class

        :param csvfile: the path to .csv file
        """
        self.csvfile = csvfile
        # Extract the header
        with open(csvfile, newline="") as f:
            reader = csv.reader(f)
            self.header = next(reader)
        # Delete the rows containing the name of the variables
        row_list = self._remove_extra_rows(csvfile)
        # Generate a pandas dataframe from row_list and convert the columns to numeric so that we can later compute the average times
        self.df = pd.DataFrame(row_list, columns=self.header).apply(pd.to_numeric, errors="ignore")

    def _remove_extra_rows(self, csvfile):
        """
        Delete from a file the rows containing the name of the variables which are written once at each repetition

        :param csvfile: the path to .csv file
        :return: the rows of the file as a list
        """
        lines = list()
        with open(csvfile, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lines.append(row)
                for field in row:
                    if field == "SplitLayer":
                        lines.remove(row)
        return lines

    def mean(self, groups):
        """
        Compute the average times

        :return: a pandas dataframe with the average value of all the numeric variables grouped by the split layer
        """
        return self.df.groupby(groups).mean()

    def export_mean_values(self, groups=["SplitLayer"]):
        """
        Generate the file with the average times
        """
        mean_values = self.mean(groups)
        self.output_file = self.csvfile.replace(".csv", "_avg.csv")
        mean_values.to_csv(self.output_file)

    def _clear_dataset(self):
        self.df = self.df[self.df["SplitLayer"] != "NO_SPLIT"]

    def reorder(self, global_order):
        """
        Reorder the rows of the average file depending on the order of the split point of the onnx model

        :param gloabal_order: a list with the order of the split points of the onnx model
        """
        order = f7(self.df["SplitLayer"].tolist())
        order = sorted(order, key=global_order.index)
        df = pd.read_csv(self.output_file)
        df = df.set_index("SplitLayer").reindex(order).reset_index()
        df.to_csv(self.output_file, index=False)

    def prepare_nextLayer_datasets(self, n_repetitions, n_train):
        """
        Generate the training and the test set for nextLayerProfiling with aMLLibrary

        :param n_repetitions: the number of repetitions to use
        :param n_train: the number of layer for training. Layers up to n-1 will be used for training, layers from n will be used for testing
        """
        # Remove the extra lines
        self._clear_dataset()

        # Extract the list with the split layers
        SplitLayer = f7(self.df["SplitLayer"].tolist())
        n_split_points = len(SplitLayer)

        # Select only the required number of repetitions
        MLdf = self.df[: n_repetitions * n_split_points]

        # Generate .csv files for training and testing the model
        train = MLdf[MLdf["SplitLayer"].isin(SplitLayer[:n_train])]
        test = self.df[self.df["SplitLayer"].isin(SplitLayer[n_train:])]
        train.to_csv("train.csv", index=False)
        test.to_csv("test.csv", index=False)

        return MLdf, n_split_points

    def compute_nextLayer_error(self, n_total_repetitions, n_split_points, n_train):
        """
        Compute the Mean Average Prediction Error for each test layer for nextLayerProfiling

        :param n_total_repetitions: the total number of repetitions in the dataset
        :param n_split_point: the number of split layers
        :param n_train: the number of layer for training
        """
        firstInfTime = np.array(self.df["1stInfTime"]).reshape(n_total_repetitions, n_split_points)
        firstInf_real = firstInfTime[:, n_train:]

        y_test = pd.read_csv("output_test/prediction.csv")
        real = np.array(y_test["real"]).reshape(n_total_repetitions, n_split_points - n_train)
        pred = np.array(y_test["pred"]).reshape(n_total_repetitions, n_split_points - n_train)

        firstInf_pred = np.cumsum(pred, axis=1)
        for i in range(n_total_repetitions):
            firstInf_pred[i] = firstInf_pred[i] + firstInfTime[i, n_train - 1]

        mape = np.mean(np.abs(firstInf_real - firstInf_pred) / firstInf_real, axis=0)
        return mape


class PlotHandler:
    """
    Manage the plot generated by files
    """

    def __init__(self, split_layers):
        """
        Initialize the object of the class

        :param split_layers: the list with the split points of the onnx model
        """
        self.split_layers = split_layers.copy()
        for i in range(len(self.split_layers)):
            self.split_layers[i] = self.split_layers[i].replace("/", "-").replace(":", "_")
        self.split_layers.append("NO_SPLIT")

    def generate_plot(self, ylim):
        n = len(self.split_layers)
        columns = 5
        rows = int(n / columns + 1)
        gs00 = GridSpec(rows, columns)

        fig = plt.figure(figsize=(20, int(8 / 3 * rows)))
        axs = []
        for i in range(rows):
            for j in range(columns):
                ax = fig.add_subplot(gs00[i, j])
                axs.append(ax)
        ax = fig.add_subplot(gs00[rows - 1, columns - 1])
        axs.append(ax)

        for row in range(rows * columns + 1):
            try:
                csvfile = self.split_layers[row] + "_avg.csv"
                df = pd.read_csv(csvfile)
                features = ["1stInfTime", "networkingTime", "2ndInfTime"]

                dfC = pd.read_csv("time_table_avg.csv").tail(-1)
                infTimeC = float(dfC[dfC["SplitLayer"] == self.split_layers[row]]["1stInfTime"])
                netTimeC = float(dfC[dfC["SplitLayer"] == self.split_layers[row]]["networkingTime"])

                axs[row].bar(
                    np.arange(df.shape[0]),
                    infTimeC,
                    color="blue",
                    alpha=0.5,
                    label="Odroid",
                    linewidth=1,
                    edgecolor="black",
                )
                axs[row].bar(
                    np.arange(df.shape[0]),
                    netTimeC,
                    bottom=infTimeC,
                    alpha=0.5,
                    label="1st Networking Time",
                    linewidth=1,
                    edgecolor="black",
                )
                axs[row].bar(
                    np.arange(df.shape[0]),
                    df[features[0]],
                    bottom=netTimeC + infTimeC,
                    alpha=0.5,
                    label="Laptop",
                    linewidth=1,
                    edgecolor="black",
                )
                axs[row].bar(
                    np.arange(df.shape[0]),
                    df[features[1]],
                    bottom=netTimeC + infTimeC + df[features[0]],
                    alpha=0.5,
                    label="2nd Networking Time",
                    linewidth=1,
                    edgecolor="black",
                )
                axs[row].bar(
                    np.arange(df.shape[0]),
                    df[features[2]],
                    bottom=netTimeC + infTimeC + df[features[1]] + df[features[0]],
                    alpha=0.5,
                    label="Desktop",
                    linewidth=1,
                    edgecolor="black",
                )
                # axs[1].set_ylabel("Time (s)")
                # axs[1].set_xlabel("Split Point")
                axs[row].tick_params(
                    axis="x",  # changes apply to the x-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                )  # labels along the bottom edge are off
                axs[row].set_ylim([0, ylim])
                axs[row].set_title("Splitting point odroid-laptop: %d" % row, fontsize=10)
                if row % columns == 0:
                    axs[row].set_ylabel("Time [s]")
                if self.split_layers[row] == "NO_SPLIT":
                    axs[row].set_title("Splitting point odroid-laptop: NO_SPLIT", fontsize=10)
                    axs[row].legend(bbox_to_anchor=(1, 1))
            except:
                axs[row].axis("off")
        fig.suptitle("Inference Times", fontsize=20)
        plt.savefig("infTimes")


def load_img(image_file, img_size_x, img_size_y, is_grayscale):
    """
    Load an image, scale it and convert it into a numpy array

    :param image_file: the path to the image
    :param img_size_x: the horizontal size of the image after the scaling
    :param img_size_y: the vertical size of the image after the scaling
    :param is_grayscale: true if the image is grayscale, false otherwise
    :return: a numpy array of size (img_size_x, img_size_y, 3)
    """
    img = Image.open(image_file)
    img.load()
    img = img.resize((img_size_x, img_size_y))
    return np.asarray(img, dtype="float32")


def f7(seq):
    """
    Extract a subsequence without duplicates from seq while maintaining the original order of the items

    :param seq: a list
    :return: a sublist containing all the elements of seq without duplicates
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
