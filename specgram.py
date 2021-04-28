import requests
import json
import time,sys
from requests.auth import HTTPBasicAuth
from collections import OrderedDict
from urllib.parse import urlencode
import matplotlib.pyplot as plt
import numpy as np
import optparse
import getpass 
import numpy as np
from numpy import savetxt

parser = optparse.OptionParser('usage: %prog -u <username> [-p <password> -r <minfreq,maxfreq> -t <timeresol> -f <frequency_resol>]')
parser.add_option('-u', '--user', dest = 'username',
                  type = 'string',
                  help = 'API username')
parser.add_option('-p', '--pass', dest = 'password',
                    type = 'string', help = 'API password')

parser.add_option('-r', '--range', dest = 'frange',
                    type = 'string', help = 'frequency range separated by commas')

parser.add_option('-t', '--tresol', dest = 'tresol',
                    type = 'string', help = 'time resolution')

parser.add_option('-f', '--fresol', dest = 'fresol',
                    type = 'string', help = 'frequency resolution')

(options, args) = parser.parse_args()
if not options.username:
   parser.error('Username not specified')

if not options.password:
   options.password = getpass.getpass('Password:')

# Electrosense API Credentials 
username = options.username
password = options.password

# Electrosense API
MAIN_URI ='https://electrosense.org/api'
SENSOR_LIST = MAIN_URI + '/sensor/list/'
SENSOR_AGGREGATED = MAIN_URI + '/spectrum/aggregated'

r = requests.get(SENSOR_LIST, auth = HTTPBasicAuth(username, password))

if r.status_code != 200:
    print(r.content)
    exit(-1)

slist_json = json.loads(r.content)

senlist = {}
status = [' (off)', ' (on)']

for i, sensor in enumerate(slist_json):
    print('[%d] %s (%d) - Sensing: %s' % (i, sensor['name'], sensor['serial'], sensor['sensing']))
    senlist[sensor['name'] + status[int(sensor['sensing'])]] = i

print('')
pos = int(input('Please enter the sensor: '))

print('')
print('%s (%d) - %s' % (slist_json[pos]['name'], slist_json[pos]['serial'], slist_json[pos]['sensing']))

def get_spectrum_data (sensor_id, timeBegin, timeEnd, aggFreq, aggTime, minfreq, maxfreq):
    
    params = OrderedDict([('sensor', sensor_id),
                          ('timeBegin', timeBegin),
                          ('timeEnd', timeEnd),
                          ('freqMin', int(minfreq)),
                          ('freqMax', int(maxfreq)),
                          ('aggFreq', aggFreq),
                          ('aggTime', aggTime),
                          ('aggFun','AVG')])

    r = requests.get(SENSOR_AGGREGATED, auth = HTTPBasicAuth(username, password), params = urlencode(params))

    if r.status_code == 200:
        return json.loads(r.content)
    else:
        print('Response: %d' % (r.status_code))
        return None

epoch_time = int(time.time())
# change the number of days for different data range
days = 1
timeBegin = epoch_time - days * 24 * 60 * 60
timeEnd = epoch_time

# timeBegin = epoch_time - 2 * days * 24 * 60 * 60
# timeEnd = epoch_time - days * 24 * 60 * 60

if not options.fresol:
    freqresol = int(100e3)
else:
    freqresol = int(float(options.fresol))

if not options.tresol:
    tresol = int(60)
else:
    tresol = int(float(options.tresol))

if not options.frange:
    minfreq = 50e6 
    maxfreq = 1500e6
else:
    minfreq = int(float(options.frange.split(',')[0])) 
    maxfreq = int(float(options.frange.split(',')[1])) 

response = get_spectrum_data(slist_json[pos]['serial'], timeBegin, timeEnd, freqresol, tresol, minfreq, maxfreq)
data = np.array(response['values']).astype(float)
savetxt('{}-{}-{}-{}-{}-{}-{}.csv'.format(slist_json[pos]['name'], timeBegin, timeEnd, tresol, minfreq, maxfreq, freqresol), data.T, delimiter = ',')

