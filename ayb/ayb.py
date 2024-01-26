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
                            num_omitted_ab_pairs=param2val['num_omitted_ab_pairs'])
    missing_training_words = training_corpus.create_vocab()

    test_corpus = XAYBZ(sentence_sequence_rule='massed',
                        random_seed=param2val['random_seed'],
                        num_omitted_ab_pairs=param2val['num_omitted_ab_pairs'])
    missing_test_words = test_corpus.create_vocab()

    if training_corpus.vocab_list != test_corpus.vocab_list:
        raise Exception("Training and Test Vocab Lists are not the same")

    the_model = create_model(training_corpus.vocab_list, param2val)

    performance_dict = train_model(param2val, the_model, training_corpus, test_corpus)

    return performance_dict


def train_model(train_params, model, training_corpus, test_corpus):
    performance_dict = {}
    took_sum = 0

    for i in range(train_params['num_epochs']):
        loss_mean, took = model.train_sequence(training_corpus, training_corpus.document_list, train_params)
        took_sum += took

        if i % train_params['eval_freq'] == 0:
            took_mean = took_sum / train_params['eval_freq']
            took_sum = 0

            evaluation_dict = evaluate_model(i, model, training_corpus, test_corpus, train_params, took_mean, loss_mean)
            print(evaluation_dict['output_string'])

        if i % train_params['save_freq'] == 0:
            file_name = f"e{i}.pth"
            model.save_model(train_params['save_path'], file_name)

    return performance_dict


if __name__ == "__main__":
    main()
