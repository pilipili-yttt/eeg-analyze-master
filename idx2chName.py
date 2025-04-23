import mne

name = 'ZCH'
vhdr_file = f"C:/Users/14152/Desktop/collaboration_measurement/data/{name}.vhdr"

raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

best_individual = [7, 13, 7, 5, 14, 17, 29, 31]
best_channel = [raw.ch_names[idx] for idx in best_individual]

print(best_channel)

import mne
import matplotlib.pyplot as plt

montage = mne.channels.make_standard_montage('standard_1020')

selected_channels = list(set(best_channel))

info = mne.create_info(ch_names=selected_channels, sfreq=250, ch_types="eeg")
info.set_montage(montage)

fig, ax = plt.subplots()
mne.viz.plot_sensors(info, axes=ax, show_names=True)
plt.show()
