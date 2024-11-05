from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.models.count_model import CountModel
from distributional_models.tasks.similarity_matrices import SimilarityMatrices
import copy
import pandas as pd
from pathlib import Path

training_corpus = XAYBZ(sentence_sequence_rule='random',
                        random_seed=1023,
                        ab_category_size=3,
                        num_ab_categories=2,
                        num_omitted_ab_pairs=0)
missing_training_words = training_corpus.create_vocab()

count_model = CountModel(training_corpus, training_corpus.vocab_list)

document_list = copy.deepcopy(training_corpus.document_list)
count_model.get_input_output_pairs(training_corpus, document_list)
count_model.count_matrix()
co_occurrence_df = pd.DataFrame(count_model.co_occurrence_matrix, index=count_model.vocab_list,
                                columns=count_model.vocab_list)
co_occurrence_df.to_csv(Path('../forward_co_occurance.csv'))
SimilarityMatrices.print_matrix(count_model.co_occurrence_matrix,
                                training_corpus.vocab_list, training_corpus.vocab_list)