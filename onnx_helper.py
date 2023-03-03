import os, sys
import csv, pandas
from PIL import Image
import numpy as np

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

import csv
import pandas as pd

class CsvHandler:
    def __init__(self, csvfile):
        self.csvfile = csvfile
        with open(csvfile, newline='') as f:
            reader = csv.reader(f)
            self.header = next(reader)
        row_list = self._remove_extra_rows(csvfile)
        self.df = pd.DataFrame(row_list, columns=self.header).apply(pd.to_numeric, errors="ignore")

    def _remove_extra_rows(self, csvfile):
        lines = list()
        with open(csvfile, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lines.append(row)
                for field in row:
                    if field == "1stInfTime":
                        lines.remove(row)
        return lines
    
    def mean(self):
        return self.df.groupby(["splitPoint1", "splitPoint2"]).mean()
    
    def export_mean_values(self):
        mean_values = self.mean()
        self.output_file = self.csvfile.replace(".csv", "_avg.csv")
        mean_values.to_csv(self.output_file)

    def reorder(self):
        order = f7(self.df["splitPoint2"].tolist())
        df = pd.read_csv(self.output_file)
        df = df.set_index("splitPoint2").reindex(order).reset_index()
        df.to_csv(self.output_file, index=False)

def load_img(image_file, img_size_x, img_size_y, is_grayscale):
    img = Image.open(image_file)
    img.load()
    img = img.resize((img_size_x, img_size_y))
    return np.asarray(img, dtype="float32")

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
