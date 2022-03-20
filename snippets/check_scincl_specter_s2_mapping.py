import json
import pickle
import collections
import statistics

import tqdm


def generator():
    while True:
        yield


if __name__ == '__main__':

    mapping = json.load(open('/gypsum/scratch1/bseoh/scincl_dataset/specter__s2id_to_s2orc_paper_id.json', 'r'))

    citation_count = json.load(open('/gypsum/scratch1/bseoh/scincl_dataset/train_citation_count.json', 'r'))

    mag_field = json.load(open('/gypsum/scratch1/bseoh/scincl_dataset/train_mag_field.json', 'r'))

    num_query_paper_ids_found = 0
    num_pos_paper_ids_found = 0
    num_neg_paper_ids_found = 0

    num_query_paper_ids_found_cited = 0
    num_pos_paper_ids_found_cited = 0
    num_neg_paper_ids_found_cited = 0

    query_paper_count_by_mag_field = collections.defaultdict(int)
    pos_paper_count_by_mag_field = collections.defaultdict(int)
    neg_paper_count_by_mag_field = collections.defaultdict(int)

    num_query_paper_ids_found_mag = 0
    num_pos_paper_ids_found_mag = 0
    num_neg_paper_ids_found_mag = 0

    num_both_found = 0
    num_all_found = 0

    num_both_found_cited = 0
    num_all_found_cited = 0

    num_both_found_mag = 0
    num_all_found_mag = 0

    unique_paper_ids = set()

    with open("/gypsum/scratch1/bseoh/original_data/train_shuffled.pkl", 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)
        for _ in tqdm.tqdm(generator()):
            try:
                instance = unpickler.load()

                query_paper_id = instance.fields.get('source_paper_id').metadata
                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata

                unique_paper_ids.add(query_paper_id)
                unique_paper_ids.add(pos_paper_id)
                unique_paper_ids.add(neg_paper_id)

                query_paper_id_found = False
                pos_paper_id_found = False
                neg_paper_id_found = False

                query_paper_id_found_cited = False
                pos_paper_id_found_cited = False
                neg_paper_id_found_cited = False

                query_paper_id_found_mag = False
                pos_paper_id_found_mag = False
                neg_paper_id_found_mag = False

                if len(query_paper_id) > 0 and query_paper_id in mapping.keys():
                    num_query_paper_ids_found += 1
                    query_paper_id_found = True

                    if query_paper_id in citation_count.keys():
                        num_query_paper_ids_found_cited += 1
                        query_paper_id_found_cited = True

                    if query_paper_id in mag_field.keys():
                        num_query_paper_ids_found_mag += 1
                        query_paper_id_found_mag = True

                        for f in mag_field[query_paper_id]['mag_field_of_study']:
                            query_paper_count_by_mag_field[f] += 1

                if len(pos_paper_id) > 0 and pos_paper_id in mapping.keys():
                    num_pos_paper_ids_found += 1
                    pos_paper_id_found = True

                    if pos_paper_id in citation_count.keys():
                        num_pos_paper_ids_found_cited += 1
                        pos_paper_id_found_cited = True

                    if pos_paper_id in mag_field.keys():
                        num_pos_paper_ids_found_mag += 1
                        pos_paper_id_found_mag = True

                        for f in mag_field[pos_paper_id]['mag_field_of_study']:
                            pos_paper_count_by_mag_field[f] += 1

                if len(neg_paper_id) > 0 and neg_paper_id in mapping.keys():
                    num_neg_paper_ids_found += 1
                    neg_paper_id_found = True

                    if neg_paper_id in citation_count.keys():
                        num_neg_paper_ids_found_cited += 1
                        neg_paper_id_found_cited = True

                    if neg_paper_id in mag_field.keys():
                        num_neg_paper_ids_found_mag += 1
                        neg_paper_id_found_mag = True

                        for f in mag_field[neg_paper_id]['mag_field_of_study']:
                            neg_paper_count_by_mag_field[f] += 1

                if pos_paper_id_found and neg_paper_id_found:
                    num_both_found += 1

                    if pos_paper_id_found_cited and neg_paper_id_found_cited:
                        num_both_found_cited += 1

                    if pos_paper_id_found_mag and neg_paper_id_found_mag:
                        num_both_found_mag += 1

                    if query_paper_id_found:
                        num_all_found += 1

                        if pos_paper_id_found_cited and neg_paper_id_found_cited and query_paper_id_found_cited:
                            num_all_found_cited += 1

                        if pos_paper_id_found_mag and neg_paper_id_found_mag and query_paper_id_found_mag:
                            num_all_found_mag += 1

            except EOFError:
                break

    print("unique paper ids", str(len(unique_paper_ids)))
    print("num_query_paper_ids_found", str(num_query_paper_ids_found))
    print("num_pos_paper_ids_found", str(num_pos_paper_ids_found))
    print("num_neg_paper_ids_found", str(num_neg_paper_ids_found))
    print("num_query_paper_ids_found_cited", str(num_query_paper_ids_found_cited))
    print("num_pos_paper_ids_found_cited", str(num_pos_paper_ids_found_cited))
    print("num_neg_paper_ids_found_cited", str(num_neg_paper_ids_found_cited))
    print("num_query_paper_ids_found_mag", str(num_query_paper_ids_found_mag))
    print("num_pos_paper_ids_found_mag", str(num_pos_paper_ids_found_mag))
    print("num_neg_paper_ids_found_mag", str(num_neg_paper_ids_found_mag))
    print("num_both_found", str(num_both_found))
    print("num_all_found", str(num_all_found))
    print("num_both_found_cited", str(num_both_found_cited))
    print("num_all_found_cited", str(num_all_found_cited))
    print("num_both_found_mag", str(num_both_found_mag))
    print("num_all_found_mag", str(num_all_found_mag))
