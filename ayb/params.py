param2requests = {
    'num_epochs': [1],
    'eval_freq': [1],
    # 'save_freq': [1],
    'num_models': [1],
    'sentence_sequence_rule': ['random'],
    'num_omitted_ab_pairs': [1],
    'activation_function': ['tanh'],
    # 'optimizer': ['sgd'],


    # Params combination works for Transformer

    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random        16/1/16/4           0.00075         2000
    #      0            random        16/1/16/4           0.00025         2000
    # 0.0005

    'model_type': ['transformer'],
    'sequence_length': [4],
    'batch_size': [1],
    'learning_rate': [0.0005],
    'transformer_embedding_size': [4],
    'transformer_num_heads': [2],
    'transformer_attention_size': [4],
    'transformer_hidden_size': [4],
    'weight_decay': [0.0],
    'momentum': [0.0],

    # Params combination works for W2V
    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random           16               0.025          200
    #      0            random           16               0.025          200
    #      0            random           03               0.025          100
    #      0            random           04               0.001          100

    # W2V Params
    # 'model_type': ['w2v'],
    # 'w2v_embedding_size': [0],
    # 'w2v_hidden_size': [4],
    # 'corpus_window_size': [4],
    # 'corpus_window_direction': ['forward'],
    # 'learning_rate': [0.001],
    # 'sequence_length': [1],
    # 'batch_size': [1],
    # 'weight_decay': [0],

    # Params combination works for SRN
    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random           16           0.0005 AdamW       2000
    #      0            random           16           0.0001 AdamW       2000
    #      0            random           6, 5         0.0001 AdamW       2000
    #      0            random           16           0.005  sgd         1000
    #      0            random           12           0.0005/0m sgd      1000
    #      0            random           8            0.005/0m sgd,0.8,0.5      1000
    #      0            random           8            0.005/0m sgd,0.8,0.5      1000
    #      0            random           4            0.0005/0m sgd,0.8,0.5     10000
    #      0            random           4            0.001 AdamW        2000
    # 'model_type': ['srn'],
    # 'rnn_embedding_size': [0],
    # 'rnn_hidden_size': [4],
    # 'learning_rate': [0.00075],
    # 'sequence_length': [4],
    # 'batch_size': [1],
    # # 'weight_init_hidden': [0.8],
    # # 'weight_init_linear': [0.5],
    # 'weight_decay': [0],
    # 'momentum': [0]

    # Params combination works for LSTM
    #     Omit      random/massed    hidden_size      learning rate     epochs
    #      1            random           16               0.00025?       2000
    #      0            random           16               0.0001         2000
    #      0            random           8                0.0002         1000/5
    #      0            random           5                0.005          2000
    #      0            random           4                0.001          2000
    # 'model_type': ['lstm'],
    # 'rnn_embedding_size': [0],
    # 'rnn_hidden_size': [4],
    # 'learning_rate': [0.001],
    # 'sequence_length': [4],
    # 'batch_size': [1],
    # 'weight_decay': [0],
}

param2default = {
    # General Params
    'random_seed': None,
    'device': 'cpu',

    # Corpus Params
    'num_ab_categories': 2,
    'ab_category_size': 3,
    'num_omitted_ab_pairs': 1,

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

    'unknown_token': '<unk>',

    # Model Params
    'model_type': 'srn',
    'gain': 1.0,
    'weight_init_hidden': 0,
    'weight_init_linear': 0,
    'momentum': 0,
    'save_path': 'models/',
    'save_freq': 100,
    'sequence_length': 4,
    'num_models': 1,
    'reset_hidden': True,

    # SRN & LSTM Params
    'rnn_embedding_size': 0,
    'rnn_hidden_size': 16,


    # W2V Params
    'w2v_embedding_size': 0,
    'w2v_hidden_size': 12,
    'corpus_window_size': 2,
    'corpus_window_direction': 'both',

    # Transformer params
    'transformer_embedding_size': 4,
    'transformer_num_heads': 2,
    'transformer_attention_size': 4,
    # 'transformer_num_layers': 3,
    'transformer_hidden_size': 4,
    'transformer_target_output': 'single_y',

    # Training Params
    'num_epochs': 1000,
    'criterion': 'cross_entropy',
    'optimizer': 'adamW',
    'activation_function': 'tanh',
    'learning_rate': 0.001,
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
    'generate_temperature': 0.8,

    # predict sequences task params
    'predict_sequences': True,

    # compare similarities task
    'compare_similarities': True,

    # evaluate states task
    'evaluate_states': False
}
