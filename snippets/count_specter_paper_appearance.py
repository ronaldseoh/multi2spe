import json
import pickle
import collections
import statistics

import tqdm


def generator():
    while True:
        yield


if __name__ == '__main__':
    occurrence_count = collections.defaultdict(int)

    with open("/gypsum/scratch1/bseoh/original_data/val_shuffled.pkl", 'rb') as f_in:
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

    with open('/gypsum/scratch1/bseoh/original_data/popularity_count.json', 'w') as popularity_count_file:
        json.dump(occurrence_count, popularity_count_file)

    print("Max occurrence:", str(statistics.max(occurrence_count.values())))
    print("Min occurrence:", str(statistics.min(occurrence_count.values())))
    print("Mean occurrence:", str(statistics.mean(occurrence_count.values())))
