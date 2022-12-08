# Copyright (C) 2015  Niklas Rosenstein, MIT License
# Last modified by Yi Jui Lee (August 15 2015)

from __future__ import division

import time
from os.path import exists
import pandas as pd
import myo
from myo.lowlevel import stream_emg
from myo.six import print_
import pickle
import json
import os
import subprocess
import caracteristicas
from caracteristicas.feature_extraction import GROUP_1

open('Emg', 'w').close()

last_t = 0
delta_t = []
timestamp_list = []
data_list = [[],[],[],[],[],[],[],[]]
total_nrg = []

flag_initialize = True

flag_muscle_activation = False
flag_processing = False

calculated_mean = 0

sample_counter = 0
window_counter = 0

HISTERESE = 1.6

THRESHOLD_IN = 1.7

THRESHOLD_OUT = 1.8

df_myo = pd.DataFrame()

temp = []
with open('PythonVars.txt') as f:
    for val in f:
        temp.append(int(val))

samplerate = temp[0]
t_s = 1 / samplerate
print("\n\nSample rate is adjusted to " + str(samplerate) + " Hz")
print("Collecting emg data every " + str(t_s) + " seconds")

subject_number = 0 # 0...4 or all
classifier_type = 'svm' # bag, lda, rf, clf, tree
# r = "classifiers\s_" + str(subject_number) + "\\" + classifier_type + "_clf.sav"
r = "classifiers\\all\\" + classifier_type + "_clf.sav"


T = temp[1]
print("\n\nThis program will terminate in " + str(T) + " seconds\n")

classifier = pickle.load(open(r, 'rb'))

with open('best_parameters.txt') as json_file:
    best_parameters = json.load(json_file)

GROUP_NUMBER = best_parameters['best_group_number']
WINDOW_SIZE  = best_parameters['best_WINDOW_SIZE']

myo.init()
r"""
There can be a lot of output from certain data like acceleration and orientation.
This parameter controls the percent of times that data is shown.
"""

class Listener(myo.DeviceListener):
    # return False from any method to stop the Hub

    def on_connect(self, myo, timestamp):
        print_("Connected to Myo")
        myo.vibrate('short')
        myo.set_stream_emg(stream_emg.enabled)
        myo.request_rssi()
        global start
        start = time.time()

    def on_rssi(self, myo, timestamp, rssi):
        print_("RSSI:", rssi)

    def on_event(self, event):
        r""" Called before any of the event callbacks. """

    def on_event_finished(self, event):
        r""" Called after the respective event callbacks have been
        invoked. This method is *always* triggered, even if one of
        the callbacks requested the stop of the Hub. """

    def on_pair(self, myo, timestamp):
        print_('Paired')
        print_("If you don't see any responses to your movements, try re-running the program or making sure the Myo works with Myo Connect (from Thalmic Labs).")


    def on_disconnect(self, myo, timestamp):
        print_('on_disconnect')

    def on_emg(self, myo, timestamp, emg):
        global t1
        global t_s
        global r
        current = time.time()
        t1 = timestamp

        show_output('emg', emg, r)

    def on_unlock(self, myo, timestamp):
        print_('unlocked')

    def on_lock(self, myo, timestamp):
        print_('locked')

    def on_sync(self, myo, timestamp):
        print_('synced')

    def on_unsync(self, myo, timestamp):
        print_('unsynced')


def show_output(message, data, r):

    global t1

    global flag_initialize
    global flag_processing
    global flag_muscle_activation

    global timestamp_list
    global data_list

    global HISTERESE
    global THRESHOLD_IN
    global THRESHOLD_OUT

    global calculated_mean

    global sample_counter
    global window_counter

    global classifier

    sample_counter = sample_counter + 1

    if flag_initialize:
        if len(timestamp_list) < WINDOW_SIZE :
            timestamp_list.append(t1)
            for i in range(8):
                data_list[i].append(data[i]/128)
        else:
            signal_energy = []
            for i in range(8):
                temp_list = data_list[i]
                for j in range(2, len(data_list)):
                    temp_list[j] = abs(data_list[i][j-1]**2 - data_list[i][j]*data_list[i][j-2])
                signal_energy.append(temp_list)

            temp_means = [sum(x) for x in zip(*signal_energy)]


            calculated_mean = abs(sum(temp_means) / len(temp_means))


            print('Calculated Mean: ' + str(calculated_mean))

            flag_initialize = False
    else:
        if len(timestamp_list) == WINDOW_SIZE:
            timestamp_list.pop(0)
            timestamp_list.append(t1)
            for i in range(8):
                data_list[i].pop(0)
                data_list[i].append(data[i] / 128)

            current_nrg = 0
            for i in range(8):
                current_nrg = current_nrg + abs(data_list[i][len(data_list[i])-2]**2 - data_list[i][len(data_list[i])-1]*data_list[i][len(data_list[i])-3])
            current_nrg = current_nrg/8
            if flag_muscle_activation:
                if (current_nrg > THRESHOLD_OUT * calculated_mean) or flag_processing:
                    if window_counter == WINDOW_SIZE:
                        print(current_nrg, calculated_mean)
                        df_sample = pd.DataFrame()
                        for i in range(8):
                            df_sample['EMG_s'+str(i)] = data_list[i]
                        df_sample['signal'] = 1
                        if GROUP_NUMBER == 1:
                            df_car_temp = caracteristicas.feature_extraction.GROUP_1(df_sample, WINDOW_SIZE, 0.001, 'G1')
                        if GROUP_NUMBER == 2:
                            df_car_temp = caracteristicas.feature_extraction.GROUP_2(df_sample, WINDOW_SIZE, 0.001, 3, 'G1')
                        sample_input = df_car_temp['feature'].to_list()
                        prediction = classifier.predict(sample_input)
                        print(prediction[0])
                        flag_processing = False
                else:
                    flag_muscle_activation = False
                window_counter = window_counter + 1
            else:
                if current_nrg > THRESHOLD_IN * calculated_mean:
                    flag_muscle_activation = True
                    flag_processing = True
                else:
                    if current_nrg < HISTERESE * calculated_mean:
                        calculated_mean = (calculated_mean * (sample_counter - 1) + current_nrg) / (sample_counter)
                        # calculated_mean = calculated_mean+current_nrg/2
                window_counter = 0

def main():

    hub = myo.Hub()
    hub.set_locking_policy(myo.locking_policy.none)

    input("Press Enter to continue...\n")

    hub.run(1000, Listener())

    print("Running...\n")

    # Listen to keyboard interrupts and stop the
    # hub in that case.
    try:
        while hub.running:
            myo.time.sleep(0.2)
    except KeyboardInterrupt:
        print_("Quitting ...")
        hub.stop(True)


if __name__ == '__main__':
    main()
