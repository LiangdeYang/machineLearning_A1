import h5py
import numpy as np

with h5py.File('images_training.h5','r') as H:
    data = np.copy(H['data'])

with h5py.File('labels_training.h5','r') as H:
    label = np.copy(H['label'])

print(label)
print(data)
