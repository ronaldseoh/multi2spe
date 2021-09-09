import pickle
import argparse
import random

import tqdm


# Check out
# https://stackoverflow.com/questions/45808140/using-tqdm-progress-bar-in-a-while-loop
# for how to use tqdm with infinite loops
def unpickle_instances():
    try:
        yield unpickler.load()
    except EOFError:
        break


if __name__ == "__main__":
    
    random.seed(413)

    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    
    args = parser.parse_args()

    instances = []

    f_in = open(args.data_file, "rb")

    unpickler = pickle.Unpickler(f_in)

    for instance in tqdm.tqdm(unpickle_instances()):
        instances.append(instance)

    f_in.close()

    print("Reading complete. Total # of instances = ", len(instances))

    print("Shuffling and writing everything back to a new file...")

    random_iteration_order = random.sample(range(len(instances)), len(instances))

    # Ported from https://github.com/allenai/specter/blob/22af37904c1540ed870b38e4cd0120a6f6705b74/specter/data_utils/create_training_files.py#L464
    outfile = args.data_file + '_shuffled'

    with open(outfile, "wb") as f_out:
        pickler = pickle.Pickler(f_out)

        idx = 0

        for i in tqdm.tqdm(random_iteration_order):
            pickler.dump(instances[i])
            idx += 1

            # to prevent from memory blow
            if idx % 2000 == 0:
                pickler.clear_memo()
