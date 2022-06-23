import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import wandb
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.signal import resample
from tqdm import tqdm

def get_data(file_name, length):
    """
    Loads the diagnoses from the .hea file.
    """
    file_name = file_name.split(".")[0]
    with open(f"../Training_WFDB/{file_name}.hea") as f:
        data = f.readlines()
    
    
    reqd = {}
    reqd["id"] = file_name
    reqd["length"] = length
    for line in data:
        if line.startswith("#Age"):
            l = line.split(":")
            try:
                age = float(l[1].strip())
            except:
                age = np.nan
            reqd["age"] = age

        if line.startswith("#Sex"):
            l = line.split(":")
            sex = l[1].strip()
            reqd["sex"] = sex
        
        if line.startswith("#Dx"):
            l = line.split(":")
            diagnosis = l[1].strip()
            if "," in diagnosis:
                diagnosis = diagnosis.split(",")
            if not isinstance(diagnosis, list):
                diagnosis = [diagnosis]
            reqd["diagnosis"] = diagnosis
    return reqd

def make_plot(signal, filename):
    """
    Plots the signal.
    """

    figure1 = go.Figure()
    x = list(range(signal.shape[1]))
    figure1.add_trace(go.Scatter(x=x, y=signal[0], name=f"Channel 1"))

    for i in range(1, 12):
        figure1.add_trace(go.Scatter(x=x, y=signal[i], name=f"Channel {i + 1}", visible="legendonly"))
    
    figure2 = go.Figure()
    x = list(range(5000))

    resampled_signals = np.array([resample(signal[i], 5000) for i in range(12)])

    figure2.add_trace(go.Scatter(x=x, y=resampled_signals[0], name=f"Channel 1"))

    for i in range(1, 12):
        figure2.add_trace(go.Scatter(x=x, y=resampled_signals[i], name=f"Channel {i + 1}", visible="legendonly"))

    return figure1, figure2


with open("code_to_condition.pkl", "rb") as f:
    code_to_condition = pickle.load(f)

files = os.listdir("../Training_WFDB")

def fetch_row(file):
    name = file.split(".")[0]
    signal = loadmat(f"../Training_WFDB/{name}.mat")["val"]
    fig1, fig2 = make_plot(signal, name)
    row = get_data(name, signal.shape[1])

    row_ = [
        name, 
        row["age"], 
        row["sex"], 
        row["length"],
        [code_to_condition[int(d)] for d in row["diagnosis"]], 
        wandb.Html(fig1.to_html()),
        wandb.Html(fig2.to_html())
    ]

    return row_

files = [i for i in files if i.endswith(".hea")]
run = wandb.init(project="PhysioNet_Challenge", entity="timeseriesbois", name="LogResampledData")
# run.use_artifact("manan-goel/PhysioNet_Challenge/PhysioNet_Dataset:v0", type="dataset")
columns = ["id", "age", "sex", "length", "diagnosis", "signals", "resampled signals"]
table = wandb.Table(columns=columns)

for idx, file in enumerate(tqdm(files)):
    row_ = fetch_row(file)
    table.add_data(*row_)


run.log({f"resampled_data_table": table})
run.finish()
