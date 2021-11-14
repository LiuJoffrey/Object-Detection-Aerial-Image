import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import Recall
from dataset import *
import json

def write_predict_csv(predicts, data, output_path, n=10):
    outputs = []
    for predict, sample in zip(predicts, data):
        candidate_ranking = [
            {
                'candidate-id': oid,
                'confidence': score.item()
            }
            for score, oid in zip(predict, sample['option_ids'])
        ]

        candidate_ranking = sorted(candidate_ranking,
                                    key=lambda x: -x['confidence'])
        best_ids = [candidate_ranking[i]['candidate-id']
                    for i in range(n)]
        outputs.append(
            ''.join(
                ['1-' if oid in best_ids
                    else '0-'
                    for oid in sample['option_ids']])
        )

    logging.info('Writing output to {}'.format(output_path))
    with open(output_path, 'w') as f:
        f.write('Id,Predict\n')
        for output, sample in zip(outputs, data):
            f.write(
                '{},{}\n'.format(
                    sample['id'],
                    output
                )
            )





print("### Loading word dictionary ###")
with open('./new_data_preprocessing_3/new_word_dict.json', 'r') as fp:
    word_dict = json.load(fp)
print(word_dict['</s>'])

print("### Loading embedding vector ###")
f = open('./new_data_preprocessing_3/new_embedding_vectors.pkl', 'rb')
embedding_vectors = pickle.load(f)
f.close()

print("### Loading test data ###")
f = open('./new_data_preprocessing_3/new_raw_test_processed.pkl', 'rb')
test_processed = pickle.load(f)
f.close()


DA_test = DialogDataset(test_processed, padding=word_dict['</s>'], shuffle=False, n_negative=-1, n_positive=-1)

from example_predictor import ExamplePredictor
PredictorClass = ExamplePredictor
predictor = PredictorClass(
        embedding=embedding_vectors
    )
model_path = "./version19_RNN_with_att_context_complex_extend/model_Att_option_complex_extend.pkl.69"

predictor.load(model_path)
print("Load Ok")

predicts = predictor.predict_dataset(DA_test, collate_fn=DA_test.collate_fn, batch_size=125)

out_dir = "./output/"
output_path = os.path.join(out_dir,
                            'predict-{}.csv'.format("ATT_7038"))
print(predicts.shape)
write_predict_csv(predicts, test_processed, output_path)

