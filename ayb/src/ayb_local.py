from distributional_models.scripts.load_model import load_model, load_models
from distributional_models.tasks.categories import Categories
from distributional_models.tasks.states import States
from pathlib import Path
from ayb.ayb import save_data, plot_data
from ayb.src.evaluate import evaluate_model
import matplotlib.pyplot as plt

from distributional_models.scripts.visualization import plot_sub_time_series, plot_sub_dendogram, plot_sub_heatmap
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import re
import csv
import os
import math

run_path = Path('../../runs/')
# run_path = Path('../AdamW_no_omit_runs/runs/')
selected_param = 'param_001'
epoch_evaluated = [f'e{epoch}' for epoch in range(0, 2001, 100)]
params, corpus, model_dict = load_models(run_path, selected_param, num_models=30, num_epochs=21)
save_path = os.path.join(run_path, params['save_path'][9:], 'extra_eval')
os.makedirs(save_path, exist_ok=True)
model_plot_info_dict = {}
total_accuracy = 0
accuracy_dict_list = []
contrast_contribution_df_list = []
global_r_df_list = []
specific_r_df_list = []
summary_dfs = []
similarity_analysis_dfs = []
prediction_analysis_dfs = []

def modify_df(df, index_list, column_name, value):
    df.insert(0, column_name, value),
    new_index = index_list + [''] * (len(df) - 1)
    df.index = new_index


for model_index, model_list in model_dict.items():
    evaluation_dict_list = []
    model_stack_analysis_dfs = []
    model_prediction_analysis_dfs = []
    last_perfect_index = -1
    for i, model in enumerate(model_list):
        epoch = int(re.findall(r'\d+', epoch_evaluated[i])[0])
        evaluation_dict = evaluate_model(i, model, corpus, corpus, params, 0, 0)
        modify_df(evaluation_dict['sequence_predictions'].model_analysis_df,['output activation'], 'epoch', epoch)
        modify_df(evaluation_dict['input_similarity_matrices'].model_analysis_df, ['input weight similarity'], 'epoch', epoch)
        modify_df(evaluation_dict['hidden_similarity_matrices'].model_analysis_df, ['hidden weight similarity'], 'epoch',
                  epoch)
        modify_df(evaluation_dict['output_similarity_matrices'].model_analysis_df, ['output weight similarity'], 'epoch', epoch)
        model_similarity_analysis_dfs =[evaluation_dict['hidden_similarity_matrices'].model_analysis_df,
                        evaluation_dict['input_similarity_matrices'].model_analysis_df,
                        evaluation_dict['output_similarity_matrices'].model_analysis_df]
        model_stack_analysis_dfs.append(pd.concat(model_similarity_analysis_dfs, axis=0))
        model_prediction_analysis_dfs.append(evaluation_dict['sequence_predictions'].model_analysis_df)
        if evaluation_dict['sequence_predictions'].accuracy_dict['B_alt_subgroup_correct'] == 1.0:
            last_perfect_index = i
        print(evaluation_dict['output_string'])
        evaluation_dict_list.append(evaluation_dict)
    stack_analysis_df = pd.concat(model_stack_analysis_dfs, axis=0)
    stack_prediction_analysis_df = pd.concat(model_prediction_analysis_dfs, axis=0)
    similarity_analysis_dfs.append(stack_analysis_df)
    prediction_analysis_dfs.append(stack_prediction_analysis_df)
    os.makedirs(save_path+'/similarity_analysis_csvs', exist_ok=True)
    os.makedirs(save_path + '/prediction_analysis_csvs', exist_ok=True)
    stack_analysis_df.to_csv(save_path+f'/similarity_analysis_csvs/model_no.{model_index}.csv')
    stack_prediction_analysis_df.to_csv(save_path + f'/prediction_analysis_csvs/model_no.{model_index}.csv')
    sequence_prediction_df, input_category_similarity_df, output_category_similarity_df, \
    input_output_similarity_matrices = save_data(params, [evaluation_dict_list], save_path, last_perfect_index)
    if params['model_type'] == 'w2v':
        max_accuracy = sequence_prediction_df['alt_accuracy'].max()
        accuracy_dict_list.append(evaluation_dict_list[last_perfect_index]['sequence_predictions'].accuracy_dict)
        total_accuracy += max_accuracy
    else:
        max_accuracy = sequence_prediction_df['accuracy'].max()
        accuracy_dict_list.append(evaluation_dict_list[last_perfect_index]['sequence_predictions'].accuracy_dict)
        total_accuracy += max_accuracy

    best_evaluation_dict = evaluation_dict_list[last_perfect_index]
    prediction_summary_dict = {
        'Model_index': [model_index],  # Put model_index in a list
        'A_group': [best_evaluation_dict['sequence_predictions'].accuracy_dict['A_group_correct']],
        'A_subgroup': [best_evaluation_dict['sequence_predictions'].accuracy_dict['A_subgroup_correct']],
        'B_group': [best_evaluation_dict['sequence_predictions'].accuracy_dict['B_group_correct']],
        'B_subgroup': [best_evaluation_dict['sequence_predictions'].accuracy_dict['B_subgroup_correct']]
    }
    prediction_summary_df = pd.DataFrame(prediction_summary_dict)
    prediction_summary_df = prediction_summary_df.round(2)
    summary_dfs.append(prediction_summary_df)

    states_evaluation = States(model=model_list[last_perfect_index], corpus=corpus, params=params, save_path=save_path,
                               layer='hidden')
    states_evaluation.get_vocab_category_subcategory_into()
    states_evaluation.generate_contrast_pairs()
    states_evaluation.get_weights()
    states_evaluation.get_hidden_states()
    states_evaluation.average_token_hidden_state_dict['<unk>'] = states_evaluation.get_hidden_state(['<unk>'])
    hidden_matrix = np.stack([states_evaluation.average_token_hidden_state_dict[key]
                              for key in model_list[last_perfect_index].vocab_list])
    hidden_similarity_matrix = np.corrcoef(hidden_matrix)

    # global_r_score_dict = {}
    # for pair in states_evaluation.contrast_pairs:
    #     key = ''.join(pair[0]) + 'vs.' + ''.join(pair[1])
    #     global_r_score_dict[key] = states_evaluation.compute_r_score(pair[0], pair[1], 'global')
    # specific_r_score_dict = {}
    # for pair in states_evaluation.contrast_pairs:
    #     key = ''.join(pair[0]) + 'vs.' + ''.join(pair[1])
    #     specific_r_score_dict[key] = states_evaluation.compute_r_score(pair[0], pair[1], 'specific')
    # contrast_contribution_df = pd.DataFrame.from_dict(states_evaluation.token_contrast_distribution_dict, orient='index')
    # contrast_contribution_df.insert(0, 'model', model_index)
    # contrast_contribution_df_list.append(contrast_contribution_df)
    # global_r_df = pd.DataFrame.from_dict(global_r_score_dict, orient='index')
    # global_r_df.insert(0, 'model', model_index)
    # global_r_df_list.append(global_r_df)
    # specific_r_df = pd.DataFrame.from_dict(specific_r_score_dict, orient='index')
    # specific_r_df.insert(0, 'model', model_index)
    # specific_r_df_list.append(specific_r_df)
    model_plot_info_dict[model_index] = [max_accuracy, sequence_prediction_df, input_category_similarity_df,
                                         output_category_similarity_df,
                                         (input_output_similarity_matrices[0].instance_similarity_matrix,
                                          input_output_similarity_matrices[1].instance_similarity_matrix),
                                         hidden_similarity_matrix]
print(f'average_accuracy: {total_accuracy / len(model_dict)}')
final_summary_df = pd.concat(summary_dfs, ignore_index=False)
final_summary_df.to_csv(save_path+'/summary_dataframe.csv', index=True)

analysis_df_sum = sum(df.select_dtypes(include='number') for df in similarity_analysis_dfs)
# Compute the average by dividing by the number of DataFrames
analysis_df_avg = analysis_df_sum / len(similarity_analysis_dfs)
analysis_df = similarity_analysis_dfs[0].copy()  # Use one of the original DataFrames as a base
analysis_df.update(analysis_df_avg)
analysis_df = analysis_df.round(2)
analysis_df.to_csv(save_path+'/similarity_analysis_df_avg.csv')

analysis_df_sum = sum(df.select_dtypes(include='number') for df in prediction_analysis_dfs)
# Compute the average by dividing by the number of DataFrames
analysis_df_avg = analysis_df_sum / len(prediction_analysis_dfs)
analysis_df = prediction_analysis_dfs[0].copy()  # Use one of the original DataFrames as a base
analysis_df.update(analysis_df_avg)
analysis_df = analysis_df.round(2)
analysis_df.to_csv(save_path+'/prediction_analysis_df_avg.csv')
# big_contrast_contribution_df = pd.concat(contrast_contribution_df_list, ignore_index=False)
# big_global_r_df = pd.concat(global_r_df_list, ignore_index=False)
# big_specific_r_df = pd.concat(specific_r_df_list, ignore_index=False)
#
# big_contrast_contribution_df.to_csv(save_path+'/contrast_contribution.csv')
# big_global_r_df.to_csv(save_path+'/global_r.csv')
# big_specific_r_df.to_csv(save_path+'/specific_r.csv')

count = 0
for value in model_plot_info_dict.values():
    if value[0] > 0.8:
        count += 1
good_model_proportion = count / len(model_dict)
sorted_model_plot_info_dict = OrderedDict(
    sorted(model_plot_info_dict.items(), key=lambda item: item[1][0], reverse=True))
sorted_accuracy_dict_list = [accuracy_dict_list[i] for i in list(sorted_model_plot_info_dict.keys())]
best_model_index, _ = next(iter(sorted_model_plot_info_dict.items()))
ranking_file = os.path.join(save_path, 'models_ranking.csv')

with open(ranking_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(sorted_model_plot_info_dict.keys())

params, corpus, model_list = load_model(run_path, selected_param, best_model_index, epoch_evaluated)

model_type = model_list[0].model_type
Bs = [(key, value) for key, value in model_list[0].vocab_index_dict.items() if 'B' in key]
As = [(key, value) for key, value in model_list[0].vocab_index_dict.items() if 'A' in key]
ys = [(key, value) for key, value in model_list[0].vocab_index_dict.items() if 'y' in key]
if model_type == 'mlp':
    model_name = 'W2V'
    hidden_size = params['w2v_hidden_size']
elif model_type == 'transformer':
    model_name = 'Transformer'
    hidden_size = params['transformer_hidden_size']
else:
    model_name = model_type.upper()
    hidden_size = params['rnn_hidden_size']

vocab_list = model_list[0].vocab_list
vocab_index_dict = model_list[0].vocab_index_dict
vocab_category_dict = corpus.create_word_category_dict(model_list[0].vocab_index_dict)
vocab_subcategory_dict = corpus.create_word_category_dict(model_list[0].vocab_index_dict, True)
vocab_categories = Categories(instance_category_dict=vocab_category_dict,
                              instance_subcategory_dict=vocab_subcategory_dict)

hidden_list = [f'H{i + 1}' for i in range(hidden_size)]
evaluation_dict_list = []
for i, model in enumerate(model_list):
    epoch = int(re.findall(r'\d+', epoch_evaluated[i])[0])
    evaluation_dict = evaluate_model(epoch, model, corpus, corpus, params, 0, 0)
    print(evaluation_dict['output_string'])
    evaluation_dict_list.append(evaluation_dict)

sequence_prediction_df, input_category_similarity_df, output_category_similarity_df, _ = \
    save_data(params, [evaluation_dict_list], save_path)
plot_data(params, model_type, sequence_prediction_df, input_category_similarity_df,
          output_category_similarity_df, save_path)

# Number of subplots
num_subplots = len(model_dict)  # Example number, change it as needed

# Calculate the number of rows and columns to make the grid as square as possible
num_rows = math.ceil(math.sqrt(num_subplots))
num_cols = math.ceil(num_subplots / num_rows)

# Create the subplots
fig_1, axs_1 = plt.subplots(num_rows, num_cols, figsize=(12, 10))
fig_2, axs_2 = plt.subplots(num_rows, num_cols, figsize=(12, 10))
fig_3, axs_3 = plt.subplots(num_rows, num_cols, figsize=(12, 10))
fig_4, axs_4 = plt.subplots(num_rows, num_cols, figsize=(24, 20))
fig_5, axs_5 = plt.subplots(num_rows, num_cols, figsize=(24, 20))
fig_6, axs_6 = plt.subplots(num_rows, num_cols, figsize=(24, 20))
fig_7, axs_7 = plt.subplots(num_rows, num_cols, figsize=(24, 20))
fig_8, axs_8 = plt.subplots(2, 2, figsize=(8, 8))
fig_9, axs_9 = plt.subplots(2, 2, figsize=(8, 8))
fig_10, axs_10 = plt.subplots(2, 2, figsize=(8, 8))
fig_11, axs_11 = plt.subplots(num_rows, num_cols, figsize=(24, 20))

# Flatten the axs array in case of a 2D array for easier indexing
axs_1 = axs_1.flatten()
axs_2 = axs_2.flatten()
axs_3 = axs_3.flatten()
axs_4 = axs_4.flatten()
axs_11 = axs_11.flatten()
axs_5 = axs_5.flatten()
axs_6 = axs_6.flatten()
axs_7 = axs_7.flatten()
axs_8 = axs_8.flatten()
axs_8_index = 0
axs_9 = axs_9.flatten()
axs_9_index = 0
axs_10 = axs_10.flatten()
axs_10_index = 0

input_cluster_info_dict_list = []
hidden_cluster_info_dict_list = []
output_cluster_info_dict_list = []


def save_cluster_info(all_clusters, cluster_category_info_dict, cluster_subcategory_info_dict):
    all_clusters = sorted(all_clusters, key=len)
    for cluster in all_clusters:
        if len(cluster) <= 3:
            cluster_subcategory = [vocab_subcategory_dict[vocab] for vocab in cluster]
            subcategory_counter = Counter(cluster_subcategory)
            most_common_subcategory, subcategory_frequency = subcategory_counter.most_common(1)[0]
            if subcategory_frequency > 1:
                if most_common_subcategory[0] == 'y':
                    subcategory_accuracy = round(subcategory_frequency / 3, 2)
                else:
                    subcategory_accuracy = round(subcategory_frequency / 6, 2)
                if cluster_subcategory_info_dict[most_common_subcategory] < subcategory_accuracy:
                    cluster_subcategory_info_dict[most_common_subcategory] = subcategory_accuracy
        if len(cluster) <= 6:
            cluster_category = [vocab_category_dict[vocab] for vocab in cluster]
            category_counter = Counter(cluster_category)
            most_common_category, category_frequency = category_counter.most_common(1)[0]
            if most_common_category != 'y':
                if category_frequency > 1:
                    category_accuracy = round(category_frequency / 6, 2)
                    if cluster_category_info_dict[most_common_category] < category_accuracy:
                        cluster_category_info_dict[most_common_category] = category_accuracy


for i, (model_index, model_plot_info_list) in enumerate(sorted_model_plot_info_dict.items()):
    sequence_prediction_df = model_plot_info_list[1]
    input_category_similarity_df = model_plot_info_list[2]
    output_category_similarity_df = model_plot_info_list[3]
    input_similarity_matrix = model_plot_info_list[4][0]
    nan_mask = np.isnan(input_similarity_matrix)
    input_similarity_matrix[nan_mask] = 1
    output_similarity_matrix = model_plot_info_list[4][1]
    nan_mask = np.isnan(output_similarity_matrix)
    output_similarity_matrix[nan_mask] = 1
    hidden_similarity_matrix = model_plot_info_list[5]
    nan_mask = np.isnan(hidden_similarity_matrix)
    hidden_similarity_matrix[nan_mask] = 1


    if model_type == 'mlp':
        model_name = 'W2V'
    elif model_type == 'transformer':
        model_name = 'Transformer'
    else:
        model_name = model_type.upper()

    line_props = {
        # 'A_Present': ('Ay-->Present A', '#13294B', 2, 'solid'),
        'A_Legal': ('Ay-->Legal A', '#13294B', 2, 'dashed'),
        'A_Omitted': ('Ay-->Omitted A', '#13294B', 2, 'dashdot'),
        'A_Illegal': ('Ay-->Illegal A', '#13294B', 2, 'dotted'),
        # 'B_Present': ('Ay-->Present B', '#C84113', 0, 'solid'),
        'B_Legal': ('Ay-->Legal B', '#C84113', 2, 'dashed'),
        'B_Omitted': ('Ay-->Omitted B', '#C84113', 2, 'dashdot'),
        'B_Illegal': ('Ay-->Illegal B', '#C84113', 2, 'dotted'),
        'y': ('Ay-->y', '#13294B', 2, 'solid'),
    }


    def standard_error(series):
        return np.std(series, ddof=1) / np.sqrt(len(series))


    title_1 = "All Models Output Activation"
    y_label = "Output Activation"
    sequence_prediction_df = sequence_prediction_df.round(5)
    sequence_prediction_df_grouped = sequence_prediction_df.groupby(
        ['model_type', 'epoch', 'token_category', 'target_category'])
    mean_sequence_prediction_df = sequence_prediction_df_grouped.agg(
        mean_output_activation_mean=('mean_output_activation', 'mean'),
        mean_output_activation_CI=('mean_output_activation', standard_error),
        accuracy=('accuracy', 'mean'),
        alt_accuracy=('alt_accuracy', 'mean')).reset_index()
    plot_sub_time_series(axs_1[i], mean_sequence_prediction_df,
                         ['target_category'],
                         'mean_output_activation_mean',
                         'mean_output_activation_CI',
                         title_1,
                         y_label,
                         (0, 1),
                         line_props,
                         True)
    if i in [1, 5, 15, 28]:
        plot_sub_time_series(axs_10[axs_10_index], mean_sequence_prediction_df,
                             ['target_category'],
                             'mean_output_activation_mean',
                             'mean_output_activation_CI',
                             title_1,
                             y_label,
                             (0, 1),
                             line_props,
                             True)
        axs_10_index += 1

    line_props = {
        'A, A_Legal': ('A <-> Legal A', '#13294B', 2, 'dashed'),
        'A, A_Illegal': ('A <-> Illegal A', '#13294B', 2, 'dotted'),
        'B, B_Legal': ('B <-> Legal B', '#C84113', 2, 'dashed'),
        'B, B_Illegal': ('B <-> Illegal B', '#C84113', 2, 'dotted'),
        'A, B_Legal': ('A/B <-> Legal B/A', '#000000', 2, 'dashed'),
        'A, B_Omitted': ('A/B <-> Omitted B/A', '#000000', 2, 'dashdot'),
        'A, B_Illegal': ('A/B <-> Illegal B/A', '#000000', 2, 'dotted'),
    }

    title_2 = "All Models Input Embedding Similarity"
    y_label = "Input Sim"
    input_category_similarity_grouped = input_category_similarity_df.groupby(
        ['model_type', 'epoch', 'token_category', 'target_category'])
    input_category_similarity_df = input_category_similarity_grouped.agg(
        mean_similarity_mean=('category_similarity', 'mean'),
        mean_similarity_CI=('category_similarity', 'std')).reset_index()
    plot_sub_time_series(axs_2[i], input_category_similarity_df,
                         ['token_category', 'target_category'],
                         'mean_similarity_mean',
                         'mean_similarity_CI',
                         title_2,
                         y_label,
                         (-1, 1),
                         line_props,
                         False)

    title_3 = "All Models Output Embedding Similarity"
    y_label = "Output Sim"
    output_category_similarity_grouped = output_category_similarity_df.groupby(
        ['model_type', 'epoch', 'token_category', 'target_category'])
    output_category_similarity_df = output_category_similarity_grouped.agg(
        mean_similarity_mean=('category_similarity', 'mean'),
        mean_similarity_CI=('category_similarity', 'std')).reset_index()
    plot_sub_time_series(axs_3[i], output_category_similarity_df,
                         ['token_category', 'target_category'],
                         'mean_similarity_mean',
                         'mean_similarity_CI',
                         title_3,
                         y_label,
                         (-1, 1),
                         line_props,
                         False)
    input_distance_matrix = np.sqrt(2 * (1 - input_similarity_matrix[1:, 1:]))
    # input_distance_matrix = 1 - np.abs(input_similarity_matrix[1:, 1:])
    title_4 = "All Models Input Dendrogram"
    _, all_clusters = plot_sub_dendogram(axs_4[i], input_distance_matrix, vocab_list[1:])
    cluster_index_dict = {'index': i + 1}
    input_cluster_subcategory_info_dict = {subcategory: 0 for subcategory in vocab_categories.subcategory_index_dict
                                           if subcategory not in ['Other', '.']}
    input_cluster_category_info_dict = {category: 0 for category in vocab_categories.category_index_dict
                                        if category not in ['Other', '.', 'y']}
    save_cluster_info(all_clusters, input_cluster_category_info_dict, input_cluster_subcategory_info_dict)
    input_cluster_info_dict = {**cluster_index_dict, **input_cluster_subcategory_info_dict,
                               **input_cluster_category_info_dict}
    input_cluster_info_dict_list.append(input_cluster_info_dict)

    hidden_distance_matrix = np.sqrt(2 * (1 - hidden_similarity_matrix[1:, 1:]))
    _, all_clusters = plot_sub_dendogram(axs_11[i], hidden_distance_matrix, vocab_list[1:])
    hidden_cluster_subcategory_info_dict = {subcategory: 0 for subcategory in vocab_categories.subcategory_index_dict
                                           if subcategory not in ['Other', '.']}
    hidden_cluster_category_info_dict = {category: 0 for category in vocab_categories.category_index_dict
                                        if category not in ['Other', '.', 'y']}
    save_cluster_info(all_clusters, hidden_cluster_category_info_dict, hidden_cluster_subcategory_info_dict)
    hidden_cluster_info_dict = {**cluster_index_dict, **hidden_cluster_subcategory_info_dict,
                               **hidden_cluster_category_info_dict}
    hidden_cluster_info_dict_list.append(hidden_cluster_info_dict)

    output_distance_matrix = np.sqrt(2 * (1 - output_similarity_matrix[1:, 1:]))
    # output_distance_matrix = 1 - np.abs(output_similarity_matrix[1:, 1:])
    title_5 = "All Models Output Dendrogram"
    _, all_clusters = plot_sub_dendogram(axs_5[i], output_distance_matrix, vocab_list[1:])
    sorted_indices = np.argsort(vocab_list)

    if i in [1, 5, 15, 28]:
        plot_sub_dendogram(axs_8[axs_8_index], output_distance_matrix, vocab_list[1:])
        plot_sub_heatmap(axs_9[axs_9_index], output_similarity_matrix, sorted_indices, vocab_list)
        axs_8_index += 1
        axs_9_index += 1

    output_cluster_subcategory_info_dict = {subcategory: 0 for subcategory in vocab_categories.subcategory_index_dict
                                            if subcategory not in ['Other', '.']}
    output_cluster_category_info_dict = {category: 0 for category in vocab_categories.category_index_dict
                                         if category not in ['Other', '.', 'y']}
    save_cluster_info(all_clusters, output_cluster_category_info_dict, output_cluster_subcategory_info_dict)
    output_cluster_info_dict = {**cluster_index_dict, **output_cluster_subcategory_info_dict,
                                **output_cluster_category_info_dict}
    output_cluster_info_dict_list.append(output_cluster_info_dict)

    title_6 = "All Models Input Heatmap"

    plot_sub_heatmap(axs_6[i], input_similarity_matrix, sorted_indices, vocab_list)
    title_7 = "All Models Output Heatmap"
    plot_sub_heatmap(axs_7[i], output_similarity_matrix, sorted_indices, vocab_list)

input_cluster_info_file = os.path.join(save_path, 'input_cluster_info.csv')
headers = input_cluster_info_dict_list[0].keys()
with open(input_cluster_info_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    # Write the header
    writer.writeheader()

    # Write the data
    writer.writerows(input_cluster_info_dict_list)

hidden_cluster_info_file = os.path.join(save_path, 'hidden_cluster_info.csv')
headers = hidden_cluster_info_dict_list[0].keys()
with open(hidden_cluster_info_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    # Write the header
    writer.writeheader()

    # Write the data
    writer.writerows(hidden_cluster_info_dict_list)

output_cluster_info_file = os.path.join(save_path, 'output_cluster_info.csv')
headers = output_cluster_info_dict_list[0].keys()
with open(output_cluster_info_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    # Write the header
    writer.writeheader()

    # Write the data
    writer.writerows(output_cluster_info_dict_list)

accuracy_info_file = os.path.join(save_path, 'accuracy_info.csv')
headers = accuracy_dict_list[0].keys()
with open(accuracy_info_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    # Write the header
    writer.writeheader()

    # Write the data
    writer.writerows(sorted_accuracy_dict_list)

fig_1.tight_layout()
fig_1.savefig(os.path.join(save_path,
                           title_1 + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_1)
fig_2.tight_layout()
fig_2.savefig(os.path.join(save_path,
                           title_2 + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_2)
fig_3.tight_layout()
fig_3.savefig(os.path.join(save_path,
                           title_3 + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_3)
fig_4.tight_layout()
fig_4.savefig(os.path.join(save_path,
                           title_4 + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_4)
fig_5.tight_layout()
fig_5.savefig(os.path.join(save_path,
                           title_5 + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_5)

fig_6.tight_layout()
fig_6.savefig(os.path.join(save_path,
                           title_6 + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_6)
fig_7.tight_layout()
fig_7.savefig(os.path.join(save_path,
                           title_7 + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_7)

fig_8.tight_layout()
fig_8.savefig(os.path.join(save_path,
                           'Output Dendrogram Examples' + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_8)

fig_9.tight_layout()
fig_9.savefig(os.path.join(save_path,
                           'Output Heatmap Examples' + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_9)

fig_10.tight_layout()
fig_10.savefig(os.path.join(save_path,
                            'Output Activation Examples' + ' num_omit ' + str(params['num_omitted_ab_pairs']) + '.png'))
plt.close(fig_10)
