import pandas as pd 
import numpy as np
import time
import math
import plotly.graph_objects as go

filename = 'WNCG_UT-1583003326-1583435326-60-877500000-882500000-100000' 
df = pd.read_csv(filename + '.csv', delimiter = ',', header = None) 
data = np.array(df)

print(np.nanmax(data))
print(np.nanmin(data))
print(np.nanmean(data))
print(np.nanmedian(data))

name = filename.split('-')
t_start = name[1]
t_end = name[2]
t_resolution = name[3]
f_start = name[4]
f_end = name[5]
f_resolution = name[6]

y = []
for i in range(int(f_start), int(f_end) + 1, int(f_resolution)):
	y.append(str(np.round(i / 10 ** 6, 2)) + 'MHz')

x = [] 
for i in range(int(t_start), int(t_end) + 1, int(t_resolution)):
	x.append(time.strftime('%Y-%m-%d %H:%M', time.localtime(i)))

data_min = np.nanmin(data)
data_max = np.nanmax(data)
new_max, new_min = 1, 0
data = (data - data_min) * (new_max - new_min) / (data_max - data_min) + new_min

# Pulsed Complex Sinusoid
F_c1 = 10
F_c2 = 50
Fs = 1 / 60
ts = 2500
te = 3000

for i in range(ts, te):
	index = np.random.randint(F_c1, F_c2, 30)
	for j in index:
		f = j + 8775
		if np.cos(2 * math.pi * i * f / Fs) > 0:
			data[j, i] += 1
		else:
			data[j, i] += -1

# Sinc pulse
for i in range(5, 10):
	Fs = 1 / 60
	Fc = 8775 + i
	ts = 0
	te = 1000
	for j in range(ts, te):
		data[i, j] += 10 * np.sinc(2 * 10 ** -7 * (j - (te + ts) / 2) * Fc / Fs)

# Non-Linear Compression
for i in range(10, 15):
	for j in range(1000, 2000):
		data[i, j] +=  2 * data[i, j] - data[i, j] ** 3

# Pulsed Chirp Events
F_c1 = 15
F_c2 = 35
Fs = 1 / 60
ts = 2000
te = 2500
slope = (F_c2 - F_c1) / (te - ts)
for i in range(ts, te):
	for j in range(int(F_c1 + slope * (i - ts)), int(F_c1 + slope * (i - ts)) + 3):
		f = j + 8775
		if np.cos(2 * math.pi * i * f / Fs) > 0:
			data[j, i] += 3
		else:
			data[j, i] += 3

# plot heatmap
plot_data = go.Heatmap(z = data,
                       x = x,
                       y = y,
                       zmin = 0,
                       zmax = 1.5,
                       colorscale ='Bluered',
                       hoverongaps = False)
layout = go.Layout(title = name[0])
fig = go.Figure(data = plot_data, layout = layout)
fig.show()
