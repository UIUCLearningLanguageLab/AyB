import time
import torch
from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.tasks.categories import Categories
from distributional_models.tasks.cohyponym_task import CohyponymTask
from distributional_models.tasks.classifier import classify
from distributional_models.models.lstm import SimpleLSTM
import torch.optim as optim


def main():
    import params
    ayb(params.param2default)


def ayb(param2val):

    the_corpus = create_ayb_corpus(param2val)
    the_categories = init_categories(the_corpus.word_category_dict)
    the_model = init_model(the_corpus,
                           param2val['sequence_length'],
                           param2val['embedding_size'],
                           param2val['hidden_layer_info_list'],
                           param2val['weight_init'],
                           param2val['device'],
                           param2val['criterion'])

    performance_dict = train_model(the_corpus, the_model, the_categories, param2val)
    # performance_dict = {}
    return performance_dict


def init_categories(category_dict):
    the_categories = Categories()
    the_categories.create_from_instance_category_dict(category_dict)
    return the_categories


def create_ayb_corpus(param2val):
    the_corpus = XAYBZ(sentence_sequence_rule=param2val['sentence_sequence_rule'],
                       random_seed=param2val['random_seed']
)
    the_corpus.create_word_category_dict()
    missing_words = the_corpus.create_vocab()
    return the_corpus


def init_model(corpus, block_size, embedding_size, hidden_layer_info_list, weight_init, device, criterion):
    model = SimpleLSTM(corpus,
                       block_size,
                       embedding_size,
                       hidden_layer_info_list,
                       weight_init,
                       criterion,
                       device=device)
    return model


def cohyponym_task(the_categories, num_thresholds):
    start_time = time.time()
    the_cohyponym_task = CohyponymTask(the_categories, num_thresholds=num_thresholds)
    mean_ba = the_cohyponym_task.run_cohyponym_task()
    took = time.time() - start_time
    return the_cohyponym_task, mean_ba, took


def classifier_task(the_categories, classifier_hidden_sizes, test_proportion, classifier_epochs,
                    classifier_lr):
    start_time = time.time()
    train_df_list, test_df_list = classify(the_categories, classifier_hidden_sizes, test_proportion=test_proportion,
                                           num_epochs=classifier_epochs, learning_rate=classifier_lr)
    train_acc_0 = train_df_list[0]['correct'].mean()
    train_acc_1 = train_df_list[1]['correct'].mean()
    train_acc_final = train_df_list[-1]['correct'].mean()
    test_acc_0 = test_df_list[0]['correct'].mean()
    test_acc_1 = test_df_list[1]['correct'].mean()
    test_acc_final = test_df_list[-1]['correct'].mean()

    took = time.time() - start_time
    return train_acc_0, train_acc_1, train_acc_final, test_acc_0, test_acc_1, test_acc_final, took


def train_model(corpus, model, the_categories, train_params):

    performance_dict = {}
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(train_params['num_epochs']):
        eval_loss = 0
        eval_tokens = 0
        eval_took = 0

        start_time = time.time()
        # TODO corpus document shuffling

        corpus_index_list = corpus.flatten_corpus_lists(corpus.document_list)

        corpus.x_list, corpus.y_list = corpus.create_index_list(corpus_index_list,
                                                                corpus.vocab_index_dict,
                                                                corpus.unknown_token,
                                                                window_size=train_params['window_size'])
        index_list = corpus.x_list + [corpus.y_list[-1]]

        sequence_list = corpus.create_sequence_lists(index_list, train_params['sequence_length'], pad_index=0)

        x_batches, y_batches = corpus.create_batches(sequence_list, train_params['batch_size'],
                                                     train_params['sequence_length'], 0)

        x_batches = [torch.tensor(x_batch, dtype=torch.long).to(model.device) for x_batch in x_batches]
        y_batches = [torch.tensor(y_batch, dtype=torch.long).to(model.device) for y_batch in y_batches]

        hidden = (torch.zeros(1, train_params['batch_size'], model.hidden_size).to(model.device),
                  torch.zeros(1, train_params['batch_size'], model.hidden_size).to(model.device))

        for x_batch, y_batch in zip(x_batches, y_batches):
            optimizer.zero_grad()
            output, hidden = model(x_batch, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())

            loss = criterion(output.view(-1, corpus.vocab_size), y_batch.view(-1))
            mask = y_batch.view(-1) != 0
            loss = (loss * mask).mean()
            loss.backward()
            optimizer.step()

            eval_loss += loss.item()
            eval_tokens += train_params['batch_size']

        training_took = time.time() - start_time

        if i % train_params['eval_freq'] == 0:

            loss_mean = eval_loss / eval_tokens
            took_mean = 1000 * (eval_took / eval_tokens)
            output_string = f"{i}  loss:{loss_mean:<7.4f}"

            weight_matrix = model.get_weights(train_params['evaluation_layer'])
            the_categories.set_instance_feature_matrix(weight_matrix, corpus.vocab_index_dict)

            if train_params['run_cohyponym_task']:
                the_cohyponym_task, mean_ba, ba_took = cohyponym_task(the_categories,
                                                                      train_params['num_thresholds'])
                output_string += f"  BA:{mean_ba:0.3f}"
            else:
                ba_took = 0

            if train_params['run_classifier_task']:

                train_0_sum = 0
                train_1_sum = 0
                train_final_sum = 0
                test_0_sum = 0
                test_1_sum = 0
                test_final_sum = 0

                for k in range(train_params['num_classifiers']):
                    the_categories.create_xy_lists()
                    train_acc_0, train_acc_1, train_acc_final, test_acc_0, test_acc_1, test_acc_final, \
                        took = classifier_task(the_categories,
                                               train_params['classifier_hidden_sizes'],
                                               train_params['test_proportion'],
                                               train_params['classifier_epochs'],
                                               train_params['classifier_lr'])
                    took_mean += took

                    train_0_sum += train_acc_0
                    train_1_sum += train_acc_1
                    train_final_sum += train_acc_final
                    test_0_sum += test_acc_0
                    test_1_sum += test_acc_1
                    test_final_sum += test_acc_final

                if train_params['num_classifiers'] > 0:
                    train_0_mean = train_0_sum / train_params['num_classifiers']
                    train_1_mean = train_1_sum / train_params['num_classifiers']
                    train_final_mean = train_final_sum / train_params['num_classifiers']
                    test_0_mean = test_0_sum / train_params['num_classifiers']
                    test_1_mean = test_1_sum / train_params['num_classifiers']
                    test_final_mean = test_final_sum / train_params['num_classifiers']

                    output_string += f"  Classify0:{train_0_mean:0.3f}-{test_0_mean:0.3f}"
                    output_string += f"  Classify1:{train_1_mean:0.3f}-{test_1_mean:0.3f}"
                    output_string += f"  ClassifyN:{train_final_mean:0.3f}-{test_final_mean:0.3f}"

            else:
                took_mean = 0

            output_string += f"  Took:{training_took:0.5f}-{ba_took:0.3f}-{took_mean:0.3f}"
            print(output_string)

    return performance_dict


if __name__ == "__main__":
    main()
