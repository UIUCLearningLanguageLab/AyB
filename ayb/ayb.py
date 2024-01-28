import copy
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from distributional_models.scripts.visualization import plot_lines
from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.scripts.create_model import create_model
from ayb.src.evaluate import evaluate_model
# from src.evaluate import evaluate_model


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

    test_corpus = copy.deepcopy(training_corpus)

    if training_corpus.vocab_list != test_corpus.vocab_list:
        raise Exception("Training and Test Vocab Lists are not the same")

    performance_dict = {}
    model_evaluation_list = []
    for i in range(param2val['num_models']):
        the_model = create_model(training_corpus.vocab_list, param2val)
        performance_dict, evaluation_dict_list = train_model(param2val, the_model, training_corpus, test_corpus)
        model_evaluation_list.append(evaluation_dict_list)

    save_data(param2val, model_evaluation_list)

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


def save_data(params, evaluation_dict_list):

    sequence_prediction_header_list = ['model_type', 'model_num', 'epoch', 'token_category', 'target_category',
                                       'mean_output_activation', 'sum_output_activation']
    category_similarity_header_list = ['model_type', 'model_num', 'epoch', 'token_category', 'target_category',
                                       'category_similarity']
    sequence_prediction_data_list = []
    category_similarity_data_list = []

    for i, model_evaluation_dict_list in enumerate(evaluation_dict_list):
        for j, evaluation_dict in enumerate(model_evaluation_dict_list):
            model_num = i + 1
            model_type = params['model_type']
            epoch = evaluation_dict['label']

            sequence_predictions = evaluation_dict['sequence_predictions']
            for k, token_category in enumerate(sequence_predictions.token_category_list):
                for l, target_category in enumerate(sequence_predictions.target_category_list):
                    sum_activation = sequence_predictions.output_activation_sum_matrix[k, l]
                    mean_activation = sequence_predictions.output_activation_mean_matrix[k, l]
                    sequence_prediction_data_list.append([model_type, model_num, epoch, token_category, target_category, mean_activation, sum_activation])

            similarity_matrices = evaluation_dict['similarity_matrices']
            for k, instance_category in enumerate(similarity_matrices.instance_category_list):
                for l, target_category in enumerate(similarity_matrices.target_category_list):
                    category_similarity = similarity_matrices.category_similarity_matrix[k, l]
                    category_similarity_data_list.append([model_type, model_num, epoch, instance_category, target_category, category_similarity])

    sequence_prediction_df = pd.DataFrame(sequence_prediction_data_list,
                                          columns=sequence_prediction_header_list)
    sequence_prediction_df = sequence_prediction_df.round(5)
    sequence_prediction_df.to_csv(os.path.join(params['save_path'], "sequence_predictions.csv"))
    sequence_prediction_df_grouped = sequence_prediction_df.groupby(['model_type', 'epoch', 'token_category', 'target_category'])
    mean_sequence_prediction_df = sequence_prediction_df_grouped.agg(
        mean_output_activation_mean=('mean_output_activation', 'mean'),
        mean_output_activation_std=('mean_output_activation', 'std')).reset_index()
    plot_time_series(mean_sequence_prediction_df)

    category_similarity_df = pd.DataFrame(category_similarity_data_list, columns=category_similarity_header_list)
    category_similarity_df = category_similarity_df.dropna()
    category_similarity_df = category_similarity_df.round(5)
    category_similarity_df.to_csv(os.path.join(params['save_path'], "category_similarity.csv"))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(category_similarity_df)

    category_similarity_grouped = category_similarity_df.groupby(
        ['model_type', 'epoch', 'token_category', 'target_category'])
    category_similarity_df = category_similarity_grouped.agg(
        mean_output_activation_mean=('category_similarity', 'mean'),
        mean_output_activation_std=('category_similarity', 'std')).reset_index()
    plot_time_series(category_similarity_df)


def plot_time_series(df):
    sns.set_style("darkgrid")

    # Create a figure and a set of subplots
    plt.figure(figsize=(10, 6))

    # Iterate over each unique combination of token_category and target_category
    for (token_cat, target_cat), group in df.groupby(['token_category', 'target_category']):
        plt.errorbar(
            group['epoch'],
            group['mean_output_activation_mean'],
            yerr=group['mean_output_activation_std'],
            label=f'{token_cat}, {target_cat}',
            capsize=3,  # Caps on error bars
            elinewidth=1,  # Width of the error bar line
            marker='o',  # Marker type
            linestyle='-'  # Line style
        )

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Mean Output Activation Mean')
    plt.title('Mean Output Activation Mean over Epochs')
    plt.legend(title='Token & Target Categories', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
