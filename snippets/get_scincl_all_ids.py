import json
import csv
#import collections
#import statistics

import tqdm


if __name__ == '__main__':

    scincl_data_file_path = "/home/bseoh/my_scratch/scincl_dataset_wol/preprocessed/scincl_dataset_wol"
    mapping = {}

    with open(specter_data_file_path, 'rb') as f_in:
        reader = csv.reader(csvfile, delimiter=',')

        for row in tqdm.tqdm(reader):
            query_paper_id = row[0]
            pos_paper_id = row[1]
            neg_paper_id = row[2]

            mapping[query_paper_id] = query_paper_id
            mapping[pos_paper_id] = pos_paper_id
            mapping[neg_paper_id] = neg_paper_id

    with open('/home/bseoh/my_scratch/scincl_dataset_wol/train_all_paper_ids.json', 'w') as all_paper_ids_file:
        json.dump(mapping, all_paper_ids_file)
