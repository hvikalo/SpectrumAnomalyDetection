#################################################################################################
# The package emulates spectrum anomaly detection on semi-experimental data (synthetic anomalies 
# embedded in real power spectral densities measurements)
#################################################################################################
# For convenience, we provide sample real data that is pre-fetched and stored in CSV files.
# To fetch new data from the Electrosense initiative, one needs to run the following line:
# python3 specgram.py -u account -p password -r 877.5e6,882.5e6
# Here "account" and "password" need to be created directly with Electrosense. The code acquires 
# PSD data with specified frequency range and time duration. To change the duration, modify lines 
# 88-90. The fetched PSD data would be stored in the current folder in the form of a CSV file and 
# can be viewed with Excel. The name of the CSV file is in the form of 
# “Sensor Name - Start Time - End Time - Time Resolution - Start Frequency - End Frequency - Frequency Resolution”. 
#################################################################################################

# Anomaly synthesis
data_generator.py

This code adds synthetic anomalies to the PSD data and saves the resulting synthetic data as two 
h5 files in the current folder. One of the files contains PSD data with anomalies while the other 
one stores labels (i.e., whether a PSD signal is an anomaly or not). Running the code also provides
a visualization of the PSD signal pre- and post- addition of the synthetic anomaly. Currently, the
code runs on two CSV files acquired by the sensors in our lab. For a different pair of CSV files
(e.g., obtained by fetching new data from Electrosense), please modify lines 19-20 of the code.
For demonstration, simply run "python3 data_generator.py".

# Convolutional auto-encoder
CNNDEC.py

This code implements a convolutional auto-encoder with a clustering and performs the anomaly detection 
task. It shows the training process and prints the accuracy achieved on training, validation and testing 
data in the terminal. For demonstration, simply run "python3 CNNDEC.py".