import time
import os
from datetime import datetime
import torch
from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.tasks.categories import Categories
from distributional_models.tasks.cohyponym_task import CohyponymTask
from distributional_models.tasks.classifier import classify
from distributional_models.models.srn import SRN
from distributional_models.models.lstm import LSTM
from distributional_models.models.mlp import MLP
from distributional_models.models.gpt import GPT


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
                           param2val['device'])
    fix_paths(param2val, "local", the_model)
    prep_output(param2val)
    performance_dict = train_model(the_corpus, the_model, the_categories, param2val)
    # performance_dict = {}
    return performance_dict


def fix_paths(train_params, run_location, model):

    if run_location == 'local':
        train_params['save_path'] = "../" + train_params['save_path']  # "models/"

    elif run_location == 'ludwig_local':
        pass

    elif run_location == 'ludwig_cluster':
        pass
        # TODO fix save_path for ludwig so it ends up in the same runs folder

    else:
        raise ValueError(f"Unrecognized run location {run_location}")

    train_params['save_path'] += f"{model.model_type}"
    train_params['save_path'] += f"_{model.vocab_size}"
    train_params['save_path'] += f"_{model.embedding_size}"
    train_params['save_path'] += f"_{model.hidden_size}"

    now = datetime.now()
    date_time_string = now.strftime("%Y%m%d_%H%M%S")
    train_params['save_path'] += f"_{date_time_string}"

    return train_params


def prep_output(train_params):
    os.mkdir(train_params['save_path'])


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


def init_model(corpus, sequence_length, embedding_size, hidden_layer_info_list, weight_init, device):
    if hidden_layer_info_list[0][0] == 'lstm':
        hidden_size = hidden_layer_info_list[0][1]
        model = LSTM(corpus, embedding_size, hidden_size, weight_init, device)
    elif hidden_layer_info_list[0][0] == 'srn':
        hidden_size = hidden_layer_info_list[0][1]
        model = SRN(corpus, embedding_size, hidden_size, weight_init, device)
    elif hidden_layer_info_list[0][0] == 'mlp':
        hidden_size = hidden_layer_info_list[0][1]
        model = MLP(corpus, embedding_size, hidden_size, weight_init, device)
    elif hidden_layer_info_list[0][0] == 'gpt':
        attention_size = hidden_layer_info_list[0][1]
        num_heads = hidden_layer_info_list[0][2]
        hidden_size = hidden_layer_info_list[0][3]
        block_size = sequence_length - 1
        model = GPT(corpus, block_size, embedding_size, num_heads, attention_size,
                    hidden_size, weight_init, device)
    else:
        raise ValueError(f"Unrecognized model type {hidden_layer_info_list[0][0]}")
    return model


def cohyponym_task(the_categories, num_thresholds):
    start_time = time.time()
    the_cohyponym_task = CohyponymTask(the_categories, num_thresholds=num_thresholds)
    mean_ba = the_cohyponym_task.run_cohyponym_task()
    took = time.time() - start_time
    return the_cohyponym_task, mean_ba, took


def classifier_task(the_categories, classifier_hidden_sizes, test_proportion, classifier_epochs,
                    classifier_lr, num_classifiers):

    start_time = time.time()

    train_0_sum = 0
    train_1_sum = 0
    train_final_sum = 0
    test_0_sum = 0
    test_1_sum = 0
    test_final_sum = 0

    for k in range(num_classifiers):
        the_categories.create_xy_lists()
        start_time = time.time()
        train_df_list, test_df_list = classify(the_categories, classifier_hidden_sizes, test_proportion=test_proportion,
                                               num_epochs=classifier_epochs, learning_rate=classifier_lr)
        train_acc_0 = train_df_list[0]['correct'].mean()
        train_acc_1 = train_df_list[1]['correct'].mean()
        train_acc_final = train_df_list[-1]['correct'].mean()
        test_acc_0 = test_df_list[0]['correct'].mean()
        test_acc_1 = test_df_list[1]['correct'].mean()
        test_acc_final = test_df_list[-1]['correct'].mean()

        train_0_sum += train_acc_0
        train_1_sum += train_acc_1
        train_final_sum += train_acc_final
        test_0_sum += test_acc_0
        test_1_sum += test_acc_1
        test_final_sum += test_acc_final

    train_0_mean = train_0_sum / num_classifiers
    train_1_mean = train_1_sum / num_classifiers
    train_final_mean = train_final_sum / num_classifiers
    test_0_mean = test_0_sum / num_classifiers
    test_1_mean = test_1_sum / num_classifiers
    test_final_mean = test_final_sum / num_classifiers

    took = time.time() - start_time

    return train_0_mean, train_1_mean, train_final_mean, test_0_mean, test_1_mean, test_final_mean, took


def prepare_batches(document_list, corpus, model, train_params):
    corpus_token_list = corpus.flatten_corpus_lists(document_list)

    corpus.x_list, corpus.y_list, corpus.index_list = corpus.create_index_list(corpus_token_list,
                                                                               corpus.vocab_index_dict,
                                                                               corpus.unknown_token,
                                                                               window_size=train_params['window_size'])

    sequence_list = corpus.create_sequence_lists(corpus.index_list, train_params['sequence_length'], pad_index=0)

    x_batches, y_batches, y_window_batches = corpus.create_batches(sequence_list, train_params['batch_size'],
                                                                   train_params['sequence_length'], 0)

    x_batches = [torch.tensor(x_batch, dtype=torch.long).to(model.device) for x_batch in x_batches]
    y_batches = [torch.tensor(y_batch, dtype=torch.long).to(model.device) for y_batch in y_batches]

    return x_batches, y_batches, y_window_batches


def evaluate_model(i, model, the_categories, corpus, train_params, training_took, loss_sum, tokens_sum):
    loss_mean = loss_sum / tokens_sum

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
        train_0_mean, \
            train_1_mean, \
            train_final_mean, \
            test_0_mean, \
            test_1_mean, \
            test_final_mean, \
            classifier_took = classifier_task(the_categories,
                                              train_params['classifier_hidden_sizes'],
                                              train_params['test_proportion'],
                                              train_params['classifier_epochs'],
                                              train_params['classifier_lr'],
                                              train_params['num_classifiers'])
        output_string += f"  Classify0:{train_0_mean:0.3f}-{test_0_mean:0.3f}"
        output_string += f"  Classify1:{train_1_mean:0.3f}-{test_1_mean:0.3f}"
        output_string += f"  ClassifyN:{train_final_mean:0.3f}-{test_final_mean:0.3f}"

    else:
        classifier_took = 0

    output_string += f"  Took:{training_took:0.2f}-{ba_took:0.2f}-{classifier_took:0.2f}"
    print(output_string)


def train_model(corpus, model, the_categories, train_params):

    performance_dict = {}
    model.train()

    model.set_optimizer(train_params['optimizer'], train_params['learning_rate'])
    model.set_criterion(train_params['criterion'])

    for i in range(train_params['num_epochs']):
        tokens_sum = 0
        loss_sum = 0

        start_time = time.time()
        # TODO corpus document shuffling

        x_batches, y_batches, y_window_batches = prepare_batches(corpus.document_list, corpus, model, train_params)

        model.init_network(train_params['batch_size'], train_params['sequence_length'])

        for x_batch, y_batch in zip(x_batches, y_window_batches):
            model.optimizer.zero_grad()
            output = model(x_batch)

            if 'lstm' in model.hidden_dict:
                model.hidden_dict['lstm'] = (model.hidden_dict['lstm'][0].detach(),
                                             model.hidden_dict['lstm'][1].detach())
            elif 'srn' in model.hidden_dict:
                model.hidden_dict['srn'] = model.hidden_dict['srn'].detach()

            loss = model.criterion(output.view(-1, corpus.vocab_size), y_batch.view(-1))
            mask = y_batch.view(-1) != 0
            loss = (loss * mask).mean()
            loss.backward()
            model.optimizer.step()

            loss_sum += loss.item()
            tokens_sum += train_params['batch_size']

        training_took = time.time() - start_time

        if i % train_params['eval_freq'] == 0:
            evaluate_model(i, model, the_categories, corpus, train_params, training_took, loss_sum, tokens_sum)
            loss_sum = 0
            tokens_sum = 0

        if i % train_params['save_freq'] == 0:
            model.save_model(train_params['save_path'] + f"/e{i}.pth")

    return performance_dict


if __name__ == "__main__":
    main()
