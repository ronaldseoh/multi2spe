import json
import csv
#import collections
#import statistics

import tqdm


if __name__ == '__main__':

    scincl_data_file_path = "train_triples.csv"
    mapping = {}

    with open(scincl_data_file_path, 'r') as f_in:
        reader = csv.reader(f_in, delimiter=',')

        for row in tqdm.tqdm(reader):
            query_paper_id = row[0]
            pos_paper_id = row[1]
            neg_paper_id = row[2]

            mapping[query_paper_id] = query_paper_id
            mapping[pos_paper_id] = pos_paper_id
            mapping[neg_paper_id] = neg_paper_id

    with open('train_all_paper_ids.json', 'w') as all_paper_ids_file:
        json.dump(mapping, all_paper_ids_file)
