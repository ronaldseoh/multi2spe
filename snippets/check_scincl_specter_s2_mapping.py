import json
import pickle


if __name__ == '__main__':

    mapping = json.load(open('/gypsum/scratch1/bseoh/scincl_dataset/specter__s2id_to_s2orc_paper_id.json', 'r'))

    num_pos_paper_ids_found = 0
    num_neg_paper_ids_found = 0

    num_both_found = 0

    with open("/gypsum/scratch1/bseoh/original_data/train_shuffled.pkl", 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)
        while True:
            try:
                instance = unpickler.load()

                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata

                if pos_paper_id in mapping.keys():
                    num_pos_paper_ids_found += 1

                if neg_paper_id in mapping.keys():
                    num_neg_paper_ids_found += 1

                if pos_paper_id in mapping.keys() and neg_paper_id in mapping.keys():
                    num_both_found += 1

    print("num_pos_paper_ids_found", str(num_pos_paper_ids_found))
    print("num_neg_paper_ids_found", str(num_neg_paper_ids_found))
    print("num_both_found", str(num_both_found))
