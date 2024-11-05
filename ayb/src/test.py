from distributional_models.corpora.xAyBz import XAYBZ

training_corpus = XAYBZ(sentence_sequence_rule='random',
                        random_seed=1023,
                        ab_category_size=3,
                        num_ab_categories=2,
                        num_omitted_ab_pairs=1)
missing_training_words = training_corpus.create_vocab()