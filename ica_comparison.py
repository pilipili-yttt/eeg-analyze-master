"""
.. _ex-ica-comp:

===========================================
Compare the different ICA algorithms in MNE
===========================================

Different ICA algorithms are fit to raw MEG data, and the corresponding maps
are displayed.

"""
# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

from time import time

import mne
from mne.datasets import sample
from mne.preprocessing import ICA

print(__doc__)

# %%
# Read and preprocess the data. Preprocessing consists of:
#
# - MEG channel selection
# - 1-30 Hz band-pass filter

vhdr_file = "C:\\Users\\14152\\Desktop\\collaboration_measurement\\data\\LZ.vhdr"

raw = mne.io.read_raw_brainvision(vhdr_file, preload=True).crop(0, 60)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)


reject = dict(mag=5e-12, grad=4000e-13)
raw.filter(1, 30, fir_design="firwin")

# %%
# Define a function that runs ICA on the raw MEG data and plots the components


def run_ica(method, fit_params=None):
    ica = ICA(
        n_components=20,
        method=method,
        fit_params=fit_params,
        max_iter="auto",
        random_state=0,
    )
    t0 = time()
    ica.fit(raw, reject=reject)
    fit_time = time() - t0
    title = "ICA decomposition using %s (took %.1fs)" % (method, fit_time)
    ica.plot_components(title=title)


# %%
# FastICA
run_ica("fastica")

# %%
# Picard
run_ica("picard")

# %%
# Infomax
run_ica("infomax")

# %%
# Extended Infomax
run_ica("infomax", fit_params=dict(extended=True))
