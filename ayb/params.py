"""
use only dictionaries to store parameters.
ludwig works on dictionaries and any custom class would force potentially unwanted logic on user.
using non-standard classes here would also make it harder for user to understand.
any custom classes for parameters should be implemented by user in main job function only.
keep interface between user and ludwig as simple as possible
"""

# will submit 3*2=6 jobs, each using a different learning rate and "configuration"
param2requests = {
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
    'sentence_sequence_rule': 'massed',

    'word_order_rule': 'fixed',
    'include_punctuation': True,

    # Model Params
    'model_type': 'lstm',
    'weight_init': 0.001,
    'save_path': 'models/',
    'save_freq': 100,
    'sequence_length': 1,

    # SRN & LSTM Params
    'rnn_embedding_size': 0,
    'rnn_hidden_size': 16,

    # W2V Params
    'w2v_embedding_size': 12,
    'w2v_hidden_size': 12,
    'corpus_window_size': 4,

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
    'learning_rate': 0.001,
    'batch_size': 1,
    'dropout_rate': 0.0,
    'l1_lambda': 0.0,
    'weight_decay': 0.0,

    # evaluation params
    'eval_freq': 20,
    'evaluation_layer': 'output',
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
    'generate_temperature': 1.0,

    # predict sequences task params
    'predict_sequences': False,

    # compare similarities task
    'compare_similarities': False
}
