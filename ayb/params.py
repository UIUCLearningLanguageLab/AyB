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
    'random_seed': None,

    'device': 'cpu',

    'num_AB_categories': 2,
    'AB_category_size': 3,

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

    'window_size': None,
    'embedding_size': 0,
    'hidden_layer_info_list': (('lstm', 16),),
    'weight_init': 0.0001,
    'sequence_length': 1,
    'criterion': 'cross_entropy',

    'num_epochs': 2000,
    'optimizer': 'adagrad',
    'learning_rate': 0.005,
    'batch_size': 1,

    'evaluation_layer': 1,
    'sequence_list': None,

    'eval_freq': 10,

    'run_cohyponym_task': True,
    'num_thresholds': 51,

    'num_classifiers': 100,
    'run_classifier_task': True,
    'classifier_hidden_sizes': (),
    'test_proportion': .1,
    'classifier_epochs': 10,
    'classifier_lr': .01
}