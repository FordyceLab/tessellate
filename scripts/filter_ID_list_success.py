import h5py
import sys

if __name__ == '__main__':
    args = sys.argv[1:]

    with h5py.File(args[0], 'r') as handle:
        data_keys = list(handle.keys())

    with open(args[1], 'r') as handle:
        accessions = [acc.strip().lower() for acc in handle.readlines()]

    for acc in accessions:
        if acc in data_keys:
            print(acc)
