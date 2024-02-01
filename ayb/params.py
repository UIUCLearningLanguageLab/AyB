param2requests = {
    'num_epochs': [2000],
    'eval_freq': [10],
    'num_models': [20],
    'sentence_sequence_rule': ['random'],
    'num_omitted_ab_pairs': [1],
    # 'optimizer': ['sgd'],


    # Params combination works for Transformer

    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random        16/1/16/4           0.00075         2000
    #      0            random        16/1/16/4           0.00025         2000


    # 'model_type': ['transformer'],
    # 'sequence_length': [4],
    # 'batch_size': [1],
    # 'learning_rate': [0.0005],
    # 'transformer_embedding_size': [16],
    # 'transformer_num_heads': [1],
    # 'transformer_attention_size': [16],
    # 'transformer_hidden_size': [4],


    # Params combination works for W2V
    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random           16               0.025          200
    #      0            random           16               0.025          200

    # W2V Params
    # 'model_type': ['w2v'],
    # 'w2v_embedding_size': [0],
    # 'w2v_hidden_size': [16],
    # 'corpus_window_size': [2],
    # 'learning_rate': [0.025],
    # 'sequence_length': [1],
    # 'batch_size': [1],

    # Params combination works for SRN
    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random           16               0.0005         2000
    #      0            random           16               0.0001         2000

    'model_type': ['srn'],
    'rnn_embedding_size': [0],
    'rnn_hidden_size': [16],
    'learning_rate': [0.0005],
    'sequence_length': [4],
    'batch_size': [1],

    # Params combination works for LSTM
    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random           16               0.00025?        2000
    #      0            random           16               0.0001         2000

    # 'model_type': ['lstm'],
    # 'rnn_embedding_size': [0],
    # 'rnn_hidden_size': [16],
    # 'learning_rate': [0.0001],
    # 'sequence_length': [4],
    # 'batch_size': [1],
}

param2default = {
    # General Params
    'random_seed': None,
    'device': 'cpu',

    # Corpus Params
    'num_ab_categories': 2,
    'ab_category_size': 3,
    'num_omitted_ab_pairs': 0,

    'x_category_size': 0,
    'y_category_size': 3,
    'z_category_size': 0,

    'min_x_per_sentence': 0,
    'max_x_per_sentence': 0,
    'min_y_per_sentence': 1,
    'max_y_per_sentence': 1,
    'min_z_per_sentence': 0,
    'max_z_per_sentence': 0,

    'document_organization_rule': 'all_pairs',
    'document_repetitions': 1,
    'document_sequence_rule': 'massed',

    'sentence_repetitions_per_document': 0,
    'sentence_sequence_rule': 'random',

    'word_order_rule': 'fixed',
    'include_punctuation': True,

    # Model Params
    'model_type': 'w2v',
    'weight_init': 0.001,
    'save_path': 'models/',
    'save_freq': 100,
    'sequence_length': 1,
    'num_models': 5,
    'reset_hidden': True,

    # SRN & LSTM Params
    'rnn_embedding_size': 0,
    'rnn_hidden_size': 16,

    # W2V Params
    'w2v_embedding_size': 0,
    'w2v_hidden_size': 12,
    'corpus_window_size': 2,

    # Transformer params
    'transformer_embedding_size': 32,
    'transformer_num_heads': 4,
    'transformer_attention_size': 8,
    # 'transformer_num_layers': 3,
    'transformer_hidden_size': 16,
    'transformer_target_output': 'single_y',

    # Training Params
    'num_epochs': 5000,
    'criterion': 'cross_entropy',
    'optimizer': 'adamW',
    'learning_rate': 0.01,
    'batch_size': 1,
    'dropout_rate': 0.0,
    'l1_lambda': 0.0,
    'weight_decay': 0.0,

    # evaluation params
    'eval_freq': 1,
    'evaluation_layer': ('input', 'output'),
    'sequence_list': None,

    # cohyponym task params
    'run_cohyponym_task': False,
    'cohyponym_similarity_metric': 'correlation',
    'cohyponym_num_thresholds': 51,
    'cohyponym_only_best_thresholds': True,

    # classifier task params
    'run_classifier_task': False,
    'num_classifiers': 1,
    'classifier_hidden_sizes': (),
    'classifier_num_folds': 10,
    'classifier_num_epochs': 30,
    'classifier_learning_rate': .05,
    'classifier_batch_size': 1,
    'classifier_criterion': 'cross_entropy',
    'classifier_optimizer': 'adam',
    'classifier_device': 'cpu',

    # generate sequence task params
    'generate_sequence': False,
    'prime_token_list': ('A1_1', 'y1'),
    'generate_sequence_length': 4,
    'generate_temperature': .01,

    # predict sequences task params
    'predict_sequences': True,

    # compare similarities task
    'compare_similarities': True
}
