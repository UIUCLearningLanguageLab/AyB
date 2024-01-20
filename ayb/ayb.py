from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.tasks.categories import Categories
from distributional_models.scripts.create_model import create_model
from distributional_models.scripts.evaluate_model import evaluate_model


def main():
    import params
    run_ayb(params.param2default, "local")


def run_ayb(param2val, run_location):

    if run_location == 'local':
        param2val['save_path'] = "../" + param2val['save_path']  # "models/"

    # create the corpus
    the_corpus = XAYBZ(sentence_sequence_rule=param2val['sentence_sequence_rule'],
                       random_seed=param2val['random_seed'])
    the_corpus.create_word_category_dict()
    missing_words = the_corpus.create_vocab()

    # create the categories
    the_categories = Categories()
    the_categories.create_from_instance_category_dict(the_corpus.word_category_dict)

    test_sequence = [['A1_1', 'y1', 'B1_2', '.']]

    # create the model
    the_model = create_model(the_corpus.vocab_list, param2val)

    performance_dict = train_model(param2val, the_corpus, the_model, the_categories, test_sequence)
    return performance_dict


def train_model(train_params, corpus, model, the_categories, test_sequence):
    performance_dict = {}
    took_sum = 0

    for i in range(train_params['num_epochs']):
        sequence = corpus.document_list
        loss_mean, took = model.train_sequence(corpus, sequence, train_params)
        took_sum += took

        if i % train_params['eval_freq'] == 0:
            took_mean = took_sum / train_params['eval_freq']
            took_sum = 0

            evaluation_dict = evaluate_model(i, model, corpus, train_params, took_mean, loss_mean,
                                             the_categories, test_sequence)
            print(evaluation_dict['output_string'])

        if i % train_params['save_freq'] == 0:
            file_name = f"e{i}.pth"
            model.save_model(train_params['save_path'], file_name)

    return performance_dict


if __name__ == "__main__":
    main()
