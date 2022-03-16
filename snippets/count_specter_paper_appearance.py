import json
import pickle
import collections

import tqdm


def generator():
    while True:
        yield


if __name__ == '__main__':
    occurrence_count = collections.defaultdict(int)

    with open("/gypsum/scratch1/bseoh/original_data/train_shuffled.pkl", 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)

        for _ in tqdm.tqdm(generator()):
            try:
                instance = unpickler.load()

                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata

                occurrence_count[pos_paper_id] += 1
                occurrence_count[neg_paper_id] += 1

            except EOFError:
                break

    print("Max occurrence:", str(max(occurrence_count.values())))
    print("Min occurrence:", str(min(occurrence_count.values())))
