from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.scripts.create_model import create_model
from src.evaluate import evaluate_model


def main():
    import params
    run_ayb(params.param2default, "local")


def run_ayb(param2val, run_location):

    if run_location == 'local':
        param2val['save_path'] = "../" + param2val['save_path']  # "models/"

    # create the corpus
    training_corpus = XAYBZ(sentence_sequence_rule=param2val['sentence_sequence_rule'],
                            random_seed=param2val['random_seed'],
                            ab_category_size=param2val['ab_category_size'],
                            num_ab_categories=param2val['num_ab_categories'],
                            num_omitted_ab_pairs=param2val['num_omitted_ab_pairs'])
    missing_training_words = training_corpus.create_vocab()

    test_corpus = XAYBZ(sentence_sequence_rule='massed',
                        random_seed=param2val['random_seed'],
                        ab_category_size=param2val['ab_category_size'],
                        num_ab_categories=param2val['num_ab_categories'],
                        num_omitted_ab_pairs=param2val['num_omitted_ab_pairs'])
    missing_test_words = test_corpus.create_vocab()

    if training_corpus.vocab_list != test_corpus.vocab_list:
        raise Exception("Training and Test Vocab Lists are not the same")

    the_model = create_model(training_corpus.vocab_list, param2val)

    performance_dict, evaluation_dict_list = train_model(param2val, the_model, training_corpus, test_corpus)

    save_data(evaluation_dict_list)

    return performance_dict


def train_model(train_params, model, training_corpus, test_corpus):
    performance_dict = {}
    took_sum = 0
    evaluation_dict_list = []

    for i in range(train_params['num_epochs']):
        loss_mean, took = model.train_sequence(training_corpus, training_corpus.document_list, train_params)
        took_sum += took

        if i % train_params['eval_freq'] == 0:
            took_mean = took_sum / train_params['eval_freq']
            took_sum = 0

            evaluation_dict = evaluate_model(i, model, training_corpus, test_corpus, train_params, took_mean, loss_mean)
            print(evaluation_dict['output_string'])
            evaluation_dict_list.append(evaluation_dict)

        if i % train_params['save_freq'] == 0:
            file_name = f"e{i}.pth"
            model.save_model(train_params['save_path'], file_name)

    return performance_dict, evaluation_dict_list


def save_data(evaluation_dict_list):

    # TODO this code should go through each dict in evaluation_dict_list and pull out the relevant data that we
    # TODO want to use to create a figure. For now, the figures we want to create are pretty simple:
    # TODO a "Output Activation given 'y'" figure with epochs as the x-axis, and different lines for the different
    # TODO conditions in the output_activation_condition_label_list below. And then a "Embedding Similarity" figure
    # TODO with epochs as the x-axis, and different lines for conditions in the similarity_condition_label_list below

    epoch_list = []
    output_activation_mean_list = []
    similarity_mean_list = []
    epoch_list = []
    output_activation_condition_label_list = ['B_Present', 'B_Legal', 'B_Illegal', 'B_Omitted', 'Other']
    similarity_condition_label_list = ['A_Present', 'A_Legal', 'A_Illegal', 'A_Omitted',
                                       'B_Present', 'B_Legal', 'B_Illegal', 'B_Omitted',
                                       'y', '.', 'Other']

    for i, eval_tuple in enumerate(evaluation_dict_list):
        pass


if __name__ == "__main__":
    main()
