from distributional_models.tasks.cohyponyms2 import Cohyponyms
from distributional_models.tasks.classifier import Classifier
from distributional_models.tasks.sequence_predictions import SequencePredictions
from distributional_models.tasks.generate import generate_sequence
from distributional_models.tasks.categories import Categories
from distributional_models.tasks.sequence_categories import SequenceCategories
from distributional_models.tasks.similarity_matrices import SimilarityMatrices


def calculate_syntagmatic_activation(activation_matrix, row_dict, column_dict):

    period_present_a = activation_matrix[row_dict['.'], column_dict['A_Present']]
    period_legal_a = activation_matrix[row_dict['.'], column_dict['A_Legal']]
    period_omitted_a = activation_matrix[row_dict['.'], column_dict['A_Omitted']]
    period_illegal_a = activation_matrix[row_dict['.'], column_dict['A_Illegal']]
    period_other = 1 - period_present_a - period_legal_a - period_omitted_a - period_illegal_a

    # TODO this needs to be made into a loop that automatically determines how many A's there are instead of hard-coding
    a_y = activation_matrix[row_dict['A1'], column_dict['y']] + activation_matrix[row_dict['A2'], column_dict['y']]
    a_other = 1 - a_y
    b_period = activation_matrix[row_dict['B1'], column_dict['.']] + activation_matrix[row_dict['B2'], column_dict['.']]
    b_other = 1 - b_period

    y_present_b = activation_matrix[row_dict['y'], column_dict['B_Present']]
    y_legal_b = activation_matrix[row_dict['y'], column_dict['B_Legal']]
    y_omitted_b = activation_matrix[row_dict['y'], column_dict['B_Omitted']]
    y_illegal_b = activation_matrix[row_dict['y'], column_dict['B_Illegal']]
    y_other = 1 - y_present_b - y_legal_b - y_omitted_b - y_illegal_b

    accuracy_mean_dict = {'period_present_a': period_present_a,
                          'period_legal_a': period_legal_a,
                          'period_omitted_a': period_omitted_a,
                          'period_illegal_a': period_illegal_a,
                          'period_other': period_other,
                          'a_y': a_y,
                          'a_other': a_other,
                          'y_present_b': y_present_b,
                          'y_legal_b': y_legal_b,
                          'y_omitted_b': y_omitted_b,
                          'y_illegal_b': y_illegal_b,
                          'y_other': y_other,
                          'b_period': b_period,
                          'b_other': b_other}

    return accuracy_mean_dict


def evaluate_model(label, model, training_corpus, test_corpus, train_params, training_took, loss_mean):

    evaluation_dict = {}

    output_string = f"{label:8}  loss:{loss_mean:<7.4f}"
    took_string = f"  Took:{training_took:0.2f}"

    # TODO this doesnt implement evaluation using hidden states
    weight_matrix = model.get_weights(train_params['evaluation_layer'])

    if 'run_cohyponym_task' in train_params:
        paradigmatic_category_dict = test_corpus.create_word_category_dict(model.vocab_index_dict)
        paradigmatic_categories = Categories(instance_category_dict=paradigmatic_category_dict)
        paradigmatic_categories.set_instance_feature_matrix(weight_matrix, test_corpus.vocab_index_dict)
        if train_params['run_cohyponym_task']:
            the_cohyponym_task = Cohyponyms(paradigmatic_categories,
                                            num_thresholds=train_params['cohyponym_num_thresholds'],
                                            similarity_metric=train_params['cohyponym_similarity_metric'],
                                            only_best_threshold=train_params['cohyponym_only_best_thresholds'])
            evaluation_dict['cohyponyms'] = the_cohyponym_task
            output_string += f" BA:{the_cohyponym_task.balanced_accuracy_mean:0.3f}-R:{the_cohyponym_task.correlation:0.3f}"
            took_string += f"-{the_cohyponym_task.took:0.2f}"

    if 'run_classifier_task' in train_params:
        paradigmatic_category_dict = test_corpus.create_word_category_dict(model.vocab_index_dict)
        paradigmatic_categories = Categories(instance_category_dict=paradigmatic_category_dict)
        paradigmatic_categories.set_instance_feature_matrix(weight_matrix, test_corpus.vocab_index_dict)
        if train_params['run_classifier_task']:
            the_classifier = Classifier(paradigmatic_categories, train_params)
            evaluation_dict['classifier'] = the_classifier
            output_string += f"  Classify0:{the_classifier.train_means[0]:0.3f}-{the_classifier.test_means[0]:0.3f}"
            output_string += f"  Classify1:{the_classifier.train_means[1]:0.3f}-{the_classifier.test_means[1]:0.3f}"
            output_string += f"  ClassifyN:{the_classifier.train_means[-1]:0.3f}-{the_classifier.test_means[-1]:0.3f}"
            took_string += f"-{the_classifier.took:0.2f}"

    if 'predict_sequences' in train_params:
        token_category_dict = training_corpus.create_word_category_dict(model.vocab_index_dict)
        token_categories = Categories(instance_category_dict=token_category_dict)
        token_categories.set_instance_feature_matrix(weight_matrix, training_corpus.vocab_index_dict)

        document_category_lists = test_corpus.assign_category_to_token(test_corpus.document_list)
        target_categories = SequenceCategories(test_corpus.document_list, test_corpus.vocab_index_dict, document_category_lists)

        if train_params['predict_sequences']:
            the_sequence_predictions = SequencePredictions(model,
                                                           test_corpus.document_list,
                                                           token_categories=token_categories,
                                                           target_categories=target_categories)
            evaluation_dict['sequence_predictions'] = the_sequence_predictions

            accuracy_mean_dict = calculate_syntagmatic_activation(the_sequence_predictions.output_activation_mean_matrix,
                                                                  token_categories.category_index_dict,
                                                                  target_categories.category_index_dict)

            output_string += f"   SeqPred:{accuracy_mean_dict['y_present_b']:0.3f}"
            output_string += f"-{accuracy_mean_dict['y_legal_b']:0.3f}"
            output_string += f"-{accuracy_mean_dict['y_omitted_b']:0.3f}"
            output_string += f"-{accuracy_mean_dict['y_illegal_b']:0.3f}"
            output_string += f"-{accuracy_mean_dict['y_other']:0.3f}"
            took_string += f"-{the_sequence_predictions.took:0.2f}"

    if train_params['compare_similarities']:
        token_category_dict = training_corpus.create_word_category_dict(model.vocab_index_dict)
        token_categories = Categories(instance_category_dict=token_category_dict)
        test_corpus.create_token_target_category_lists()
        the_sim_matrices = SimilarityMatrices(weight_matrix,
                                              test_corpus.vocab_index_dict,
                                              instance_categories=token_categories,
                                              instance_target_category_list_dict=test_corpus.token_target_category_list_dict)
        evaluation_dict['similarity_matrices'] = the_sim_matrices

    if train_params['generate_sequence']:
        generated_sequence = generate_sequence(model,
                                               train_params['prime_token_list'],
                                               train_params['generate_sequence_length'],
                                               train_params['generate_temperature'])
        output_string += f'   "{generated_sequence}"'

    evaluation_dict['output_string'] = output_string + took_string

    return evaluation_dict


def calculate_categorized_similarities(model, token_category_dict, token_target_category_dict):
    pass
