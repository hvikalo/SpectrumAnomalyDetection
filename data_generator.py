import pandas as pd 
import numpy as np
import time
import math
import plotly.graph_objects as go
import h5py

def load_data(filename):
    df = pd.read_csv(filename + '.csv', delimiter = ',', header = None) 
    data = np.array(df).T
    index = []
    
    for i in range(data.shape[0]):
        if not math.isnan(data[i, 0]):
            index.append(i)
    
    return data

data_1 = load_data('WNCG_UT-1583003326-1583435326-60-877500000-882500000-100000')
data_2 = load_data('WNCG_UT-1582571433-1583003433-60-877500000-882500000-100000')

window_len = 48
merged_data = []
for i in range(data_1.shape[0] - window_len):
    merged_data.append(data_1[i : i + window_len, :].tolist())
    
for i in range(data_2.shape[0] - window_len):
    merged_data.append(data_2[i : i + window_len, :].tolist())
    
merged_data = np.array(merged_data)
np.random.shuffle(merged_data)
# remove nan
index = []
for i in range(merged_data.shape[0]):
    if not np.isnan(merged_data[i, :, :]).any():
        index.append(i)
        
merged_data = merged_data[index, :, :]

anomaly_data = merged_data.copy()

p_anomaly = 0.5
y = np.zeros((anomaly_data.shape[0]))
t_threshold = int(anomaly_data.shape[1] / 2)
f_threshold = int(anomaly_data.shape[2] / 2)
anomaly_len = 5
for z in range(int(p_anomaly * anomaly_data.shape[0])):
    y[z] = 1
    # Pulsed Chirp Events
    while True:
        F_c1 = np.random.randint(anomaly_data.shape[2])
        F_c2 = np.random.randint(anomaly_data.shape[2])
        ts = np.random.randint(anomaly_data.shape[1])
        te = np.random.randint(anomaly_data.shape[1])
        if F_c2 - F_c1 > f_threshold and te - ts > t_threshold and F_c2 < anomaly_data.shape[2] - anomaly_len:
            break
    slope = (F_c2 - F_c1) / (te - ts)
    for i in range(ts, te + 1):
        for j in range(int(F_c1 + slope * (i - ts)), int(F_c1 + slope * (i - ts)) + anomaly_len):
            anomaly_data[z, i, j] += 3 # add 3dB

# plot heatmap
plot_data = go.Heatmap(z = anomaly_data[5000, :48, :48].T,
                       x = np.arange(anomaly_data.shape[2]),
                       y = np.arange(anomaly_data.shape[1]),
#                        zmin = 0,
#                        zmax = 6,
                       colorscale ='Bluered',
                       hoverongaps = False)
layout = go.Layout(title = 'heatmap')
fig = go.Figure(data = plot_data, layout = layout)
fig.show()

# plot heatmap
plot_data = go.Heatmap(z = merged_data[5000, :48, :48].T,
                       x = np.arange(anomaly_data.shape[2]),
                       y = np.arange(anomaly_data.shape[1]),
#                        zmin = 0,
#                        zmax = 6,
                       colorscale ='Bluered',
                       hoverongaps = False)
layout = go.Layout(title = 'heatmap')
fig = go.Figure(data = plot_data, layout = layout)
fig.show()

with h5py.File('anomaly_50.h5', 'w') as hf:
    hf.create_dataset('anomaly_50',  data = np.array(anomaly_data))
with h5py.File('label_50.h5', 'w') as hf:
    hf.create_dataset('label_50',  data = np.array(y))
