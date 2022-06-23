import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import plotly.express as px
import plotly.graph_objects as go

def get_data(file_name):
    """
    Loads the diagnoses from the .hea file.
    """
    file_name = file_name.split(".")[0]
    with open(f"artifacts/PhysioNet_Dataset:v0/{file_name}.hea") as f:
        data = f.readlines()
    
    signal = loadmat(f"artifacts/PhysioNet_Dataset:v0/{file_name}.mat")["val"]
    
    reqd = {}
    reqd["id"] = file_name
    reqd["length"] = signal.shape[1]
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

    figure = go.Figure()
    x = list(range(signal.shape[1]))
    figure.add_trace(go.Scatter(x=x, y=signal[0], name=f"Channel 1"))

    for i in range(1, 12):
        figure.add_trace(go.Scatter(x=x, y=signal[i], name=f"Channel {i + 1}", visible="legendonly"))
        
    return figure


import pickle
with open("./code_to_condition.pkl", "rb") as f:
    code_to_condition = pickle.load(f)

files = os.listdir("./Training_WFDB")

files = [i for i in files if i.endswith(".hea")]
run = wandb.init(project="PhysioNet_Challenge", entity="timeseriesbois")
run.use_artifact("timeseriesbois/PhysioNet_Challenge/PhysioNet_Dataset:v0", type="dataset")
columns = ["id", "age", "sex", "diagnosis", "signals"]
table = wandb.Table(columns=columns)

for idx, file in enumerate(tqdm(files)):
    name = file.split(".")[0]
    signal = loadmat(f"./Training_WFDB/{name}.mat")["val"]
    figure = make_plot(signal, name)
    row = get_data(name)

    row_ = [
        name, 
        row["age"], 
        row["sex"], 
        [code_to_condition[int(d)] for d in row["diagnosis"]], 
        wandb.Html(figure.to_html())
    ]
    table.add_data(*row_)


run.log({f"data_table": table})
run.finish()
