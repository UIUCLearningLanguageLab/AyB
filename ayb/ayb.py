from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.tasks.categories import Categories
from distributional_models.scripts.create_model import create_model
from distributional_models.tasks.cohyponyms2 import Cohyponyms
from distributional_models.tasks.classifier import Classifier
from distributional_models.tasks.sequence_predictions import SequencePredictions
from distributional_models.tasks.generate import generate_sequence
# from distributional_models.scripts.evaluate_model import evaluate_model


def main():
    import params
    run_ayb(params.param2default, "local")


def run_ayb(param2val, run_location):

    if run_location == 'local':
        param2val['save_path'] = "../" + param2val['save_path']  # "models/"

    # create the corpus
    training_corpus = XAYBZ(sentence_sequence_rule=param2val['sentence_sequence_rule'],
                            random_seed=param2val['random_seed'])
    missing_training_words = training_corpus.create_vocab()

    test_corpus = XAYBZ(sentence_sequence_rule='massed',
                        random_seed=param2val['random_seed'])
    missing_test_words = test_corpus.create_vocab()

    if training_corpus.vocab_list != test_corpus.vocab_list:
        raise Exception("Training and Test Vocab Lists are not the same")

    the_model = create_model(training_corpus.vocab_list, param2val)

    word_category_dict = training_corpus.create_word_category_dict(the_model.vocab_index_dict)
    sequence_target_label_list = test_corpus.assign_category_index_to_token(test_corpus.document_list)

    # create the categories
    the_categories = Categories()

    the_categories.create_from_instance_category_dict(word_category_dict)

    performance_dict = train_model(param2val, the_model, training_corpus, test_corpus, the_categories,
                                   sequence_target_label_list)

    return performance_dict


def train_model(train_params, model, training_corpus, test_corpus, the_categories, sequence_target_label_list):
    performance_dict = {}
    took_sum = 0

    for i in range(train_params['num_epochs']):
        loss_mean, took = model.train_sequence(training_corpus, training_corpus.document_list, train_params)
        took_sum += took

        if i % train_params['eval_freq'] == 0:
            took_mean = took_sum / train_params['eval_freq']
            took_sum = 0

            evaluation_dict = evaluate_model(i, model, training_corpus, test_corpus, train_params, took_mean, loss_mean,
                                             sequence_target_label_list, the_categories)
            print(evaluation_dict['output_string'])

        if i % train_params['save_freq'] == 0:
            file_name = f"e{i}.pth"
            model.save_model(train_params['save_path'], file_name)

    return performance_dict


def evaluate_model(label, model, training_corpus, test_corpus, train_params, training_took, loss_mean,
                   sequence_target_label_list, the_categories=None):

    evaluation_dict = {}

    output_string = f"{label:8}  loss:{loss_mean:<7.4f}"
    took_string = f"  Took:{training_took:0.2f}"

    # TODO this doesnt implement evaluation using hidden states
    weight_matrix = model.get_weights(train_params['evaluation_layer'])

    if the_categories is not None:
        the_categories.set_instance_feature_matrix(weight_matrix, training_corpus.vocab_index_dict)

        if 'run_cohyponym_task' in train_params:
            if train_params['run_cohyponym_task']:
                the_cohyponym_task = Cohyponyms(the_categories,
                                                num_thresholds=train_params['cohyponym_num_thresholds'],
                                                similarity_metric=train_params['cohyponym_similarity_metric'],
                                                only_best_threshold=train_params['cohyponym_only_best_thresholds'])
                evaluation_dict['cohyponyms'] = the_cohyponym_task
                output_string += f" BA:{the_cohyponym_task.balanced_accuracy_mean:0.3f}-R:{the_cohyponym_task.correlation:0.3f}"
                took_string += f"-{the_cohyponym_task.took:0.2f}"

        if 'run_classifier_task' in train_params:
            if train_params['run_classifier_task']:
                the_classifier = Classifier(the_categories, train_params)
                evaluation_dict['classifier'] = the_classifier
                output_string += f"  Classify0:{the_classifier.train_means[0]:0.3f}-{the_classifier.test_means[0]:0.3f}"
                output_string += f"  Classify1:{the_classifier.train_means[1]:0.3f}-{the_classifier.test_means[1]:0.3f}"
                output_string += f"  ClassifyN:{the_classifier.train_means[-1]:0.3f}-{the_classifier.test_means[-1]:0.3f}"
                took_string += f"-{the_classifier.took:0.2f}"

    if 'predict_sequences' in train_params:
        if train_params['predict_sequences']:
            the_sequence_predictions = SequencePredictions(model,
                                                           test_corpus.document_list,
                                                           sequence_target_label_list=sequence_target_label_list,
                                                           token_category_dict=the_categories.instance_category_dict,
                                                           target_category_index_dict=test_corpus.target_category_index_dict)
            evaluation_dict['sequence_predictions'] = the_sequence_predictions
            output_string += f" SeqPred:{the_sequence_predictions.sequence_prediction_accuracy_mean:0.3f}"
            took_string += f"-{the_sequence_predictions.took:0.2f}"


            # sequence_target_label_list = corpus.assign_category_index_to_token(test_sequence_list)
            # print("corpus target_category_index dict", corpus.target_category_index_dict)
            # the_sequence_predictions = SequencePredictions(model,
            #                                                test_sequence_list,
            #                                                sequence_target_label_list=sequence_target_label_list,
            #                                                token_category_dict=corpus.word_category_dict,
            #                                                target_category_index_dict=corpus.target_category_index_dict)
            # evaluation_dict['sequence_predictions'] = the_sequence_predictions
            # output_string += f" SeqPred:{the_sequence_predictions.sequence_prediction_accuracy_mean:0.3f}"
            # took_string += f"-{the_sequence_predictions.took:0.2f}"





    if train_params['generate_sequence']:
        generated_sequence = generate_sequence(model,
                                               train_params['prime_token_list'],
                                               train_params['generate_sequence_length'],
                                               train_params['generate_temperature'])
        output_string += f'   "{generated_sequence}"'

    evaluation_dict['output_string'] = output_string + took_string

    return evaluation_dict


if __name__ == "__main__":
    main()
