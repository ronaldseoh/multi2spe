import pickle
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('data_file')
    
    args = parser.parse_args()

    dataset_size = 0

    with open(args.data_file, 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)

        while True:
            try:
                instance = unpickler.load()
                dataset_size += 1
            except EOFError:
                break

    print(dataset_size)
