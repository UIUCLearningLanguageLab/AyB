from distributional_models.tasks.cohyponyms2 import Cohyponyms
from distributional_models.tasks.classifier import Classifier
from distributional_models.tasks.sequence_predictions import SequencePredictions
from distributional_models.tasks.generate import generate_sequence
from distributional_models.tasks.categories import Categories
from distributional_models.tasks.sequence_categories import SequenceCategories
from distributional_models.tasks.similarity_matrices import SimilarityMatrices
import copy


def evaluate_model(label, model, training_corpus, test_corpus, train_params, training_took, loss_mean):

    evaluation_dict = {'label': label}

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
            output_string += \
                f" BA:{the_cohyponym_task.balanced_accuracy_mean:0.3f}-R:{the_cohyponym_task.correlation:0.3f}"
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
        if train_params['predict_sequences']:
            token_category_dict = training_corpus.create_word_category_dict(model.vocab_index_dict)
            token_categories = Categories(instance_category_dict=token_category_dict)
            token_categories.set_instance_feature_matrix(weight_matrix, training_corpus.vocab_index_dict)

            document_category_lists = test_corpus.assign_categories_to_token_target_sequences(test_corpus.document_list)
            target_categories = SequenceCategories(test_corpus.document_list,
                                                   test_corpus.vocab_index_dict,
                                                   document_category_lists)

            token_list = copy.deepcopy(model.vocab_list)
            remove_list = []
            for token in token_list:
                if token[0] != 'y':
                    remove_list.append(token)
            for item in remove_list:
                while item in token_list:
                    token_list.remove(item)

            target_list = copy.deepcopy(model.vocab_list)
            remove_list = ['<unk>', 'y1', 'y2', 'y3', '.']
            for item in remove_list:
                while item in target_list:
                    target_list.remove(item)

            the_sequence_predictions = SequencePredictions(model,
                                                           test_corpus.document_list,
                                                           token_list=token_list,
                                                           target_list=target_list,
                                                           token_categories=token_categories,
                                                           target_categories=target_categories)
            evaluation_dict['sequence_predictions'] = the_sequence_predictions
            took_string += f"-{the_sequence_predictions.took:0.2f}"

    if train_params['compare_similarities']:
        token_category_dict = training_corpus.create_word_category_dict(model.vocab_index_dict)
        token_categories = Categories(instance_category_dict=token_category_dict)
        test_corpus.assign_categories_to_token_targets()
        instance_list = copy.deepcopy(model.vocab_list)
        remove_list = ['<unk>', 'y1', 'y2', 'y3', '.']
        for item in remove_list:
            while item in instance_list:
                instance_list.remove(item)

        the_sim_matrices = SimilarityMatrices(weight_matrix,
                                              test_corpus.vocab_index_dict,
                                              instance_list=instance_list,
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
