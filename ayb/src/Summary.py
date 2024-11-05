import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np


srn_4_path = Path('../../runs/param_001')
srn_16_path = Path('../../runs/param_002')
lstm_4_path = Path('../../runs/param_003')
lstm_16_path = Path('../../runs/param_004')
w2v_4_path = Path('../../runs/param_005')
w2v_16_path = Path('../../runs/param_006')
tf_4_path = Path('../../runs/param_007')
tf_16_path = Path('../../runs/param_008')
model_path_list = [srn_4_path, lstm_4_path, w2v_4_path, tf_4_path, srn_16_path, lstm_16_path, w2v_16_path, tf_16_path]
model_list = ['srn_4', 'lstm_4', 'w2v_4', 'tf_4', 'srn_16', 'lstm_16', 'w2v_16', 'tf_16']
model_dfs_list = []
model_accuracy_dfs_list = []
for model_path in model_path_list:
    runs = [d for d in model_path.iterdir() if d.is_dir()]
    runs.sort()
    run = runs[0]
    accuracy_df = pd.read_csv(run / 'saves/extra_eval/accuracy_info.csv')
    if (accuracy_df['B_subgroup_correct'] == 0).all():
        accuracy_df['B_subgroup_correct'] = accuracy_df['B_alt_subgroup_correct']
    if (accuracy_df['A_subgroup_correct'] == 0).all():
        accuracy_df['A_subgroup_correct'] = accuracy_df['A_alt_subgroup_correct']
    input_cluster_df = pd.read_csv(run / 'saves/extra_eval/input_cluster_info.csv')
    input_cluster_df['A sub'] = input_cluster_df['A1'] + input_cluster_df['A2']
    input_cluster_df['B sub'] = input_cluster_df['B1'] + input_cluster_df['B2']
    hidden_cluster_df = pd.read_csv(run / 'saves/extra_eval/hidden_cluster_info.csv')
    hidden_cluster_df['A sub'] = hidden_cluster_df['A1'] + hidden_cluster_df['A2']
    hidden_cluster_df['B sub'] = hidden_cluster_df['B1'] + hidden_cluster_df['B2']
    output_cluster_df = pd.read_csv(run / 'saves/extra_eval/output_cluster_info.csv')
    output_cluster_df['A sub'] = output_cluster_df['A1'] + output_cluster_df['A2']
    output_cluster_df['B sub'] = output_cluster_df['B1'] + output_cluster_df['B2']
    entropy_df = pd.read_csv(run / 'saves/extra_eval/hidden_analysis_info.csv')
    entropy_df['A sub'] = entropy_df[['A1', 'A2']].mean(axis=1)
    entropy_df['B sub'] = entropy_df[['B1', 'B2']].mean(axis=1)
    model_dfs_list.append([input_cluster_df, hidden_cluster_df, output_cluster_df])
    model_accuracy_dfs_list.append(accuracy_df)


def adjust_subplot_position(ax, row_offset, col_offset, height_factor):
    pos = ax.get_position()
    new_pos = [pos.x0 + col_offset, pos.y0 + row_offset, pos.width, pos.height * height_factor]
    ax.set_position(new_pos)

fig_width = 8.5
fig_height = 5
box_fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(fig_width, fig_height))
title = 'Accuracy'
flierprops = dict(marker='o', markersize=3, linestyle='none', markeredgecolor='black')
medianprops = dict(color='black')
column_names = ['A_subgroup_correct', 'A_omit_correct',
                        'B_subgroup_correct', 'B_omit_correct']
bar_colors = ['#0072B2', '#56B4E9', '#E69F00', '#009E73']
title = 'accuracy'
y_axis_label = 'Accuracy'
ylim = (0, 1.3)
y_ticks = np.arange(0, 1.1, 0.5)
xtick_labels = ['A sub', 'A omit', 'B sub', 'B omit']
row_offsets = [0.01, 0.0]
column_offsets = [-0.01, -0.0, 0.01, 0.02, -0.01, -0.0, 0.01, 0.02]
row_index = 0
for i in range(8):
    if i == 4:
        row_index += 1
    data = model_accuracy_dfs_list[i][column_names]
    axs[row_index, i % 4].boxplot(data.values, flierprops=flierprops, medianprops=medianprops)
    axs[row_index, i % 4].set_title(f'{model_list[i]}', fontsize=10)
    axs[row_index, i % 4].set_xticks(range(1, len(xtick_labels) + 1))
    axs[row_index, i % 4].set_xticklabels(xtick_labels, rotation=90)

    if ylim is not None:
        axs[row_index, i % 4].set_ylim(ylim[0] - 0.3, ylim[1])
        axs[row_index, i % 4].set_yticks(y_ticks)
    if i % 4 == 0:
        axs[row_index, i % 4].set_ylabel(y_axis_label, fontsize=9)
    axs[row_index, i % 4].tick_params(axis='both', which='major', labelsize=8)
    means = [np.mean(d) for d in data.values.T]
    y_min, y_max = axs[row_index, i % 4].get_ylim()
    annot_offset = (y_max - y_min) * 0.15
    for k, mean in enumerate(means, start=1):
        if not np.isnan(mean):
            axs[row_index, i % 4].text(k, y_max - annot_offset, f'{mean:.2f}', horizontalalignment='center',
                                       color='black', fontsize=6)
            if i < 4:
                adjust_subplot_position(axs[row_index, i % 4], row_offsets[0], column_offsets[i], 0.95)
            else:
                adjust_subplot_position(axs[row_index, i % 4], row_offsets[1], column_offsets[i], 0.95)

    box_fig.text(0.5, 0.95, 'Prediction Accuracy', ha='center', va='center', fontsize=13, fontweight='bold')

plt.show()




# Define the figure size (width x height in inches)
fig_width = 8.5
fig_height = 10

# Create the main figure with specified size
box_fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(fig_width, fig_height))
title_list = ['Accuracy', 'Input Similarity Clustering', 'Output Similarity Clustering', 'Entropy of hidden state']
row_index = 0
# flierprops = dict(marker='o', markersize=3, linestyle='none', markeredgecolor='black')
# medianprops = dict(color='black')
for i in range(3):
    if i == 10:
        column_names = ['A_subgroup_correct', 'A_omit_correct',
                        'B_subgroup_correct', 'B_omit_correct']
        bar_colors = ['#0072B2', '#56B4E9', '#E69F00', '#009E73']
        title = 'accuracy'
        y_axis_label = 'Accuracy'
        ylim = (0, 1.3)
        y_ticks = np.arange(0, 1.1, 0.5)
        xtick_labels = ['A sub', 'A omit', 'B sub', 'B omit']
    elif 3 > i >= 0:
        ylim = (0, 1.3)
        y_ticks = np.arange(0, 1.1, 0.5)
        y_axis_label = 'Cluster Size'
        xtick_labels = ['A', 'A sub', 'y', 'B', 'B sub']
        column_names = ['A', 'A sub', 'y',
                        'B', 'B sub']
        bar_colors = ['#0072B2', '#56B4E9', '#E69F00', '#009E73', '#B2DF8A']
        if i == 0:
            rep_type = 'input'
            title = 'input similarity clustering'
        elif i == 1:
            rep_type = 'hidden'
            title = 'hidden similarity clustering'
        elif i == 2:
            rep_type = 'output'
            title = 'output similarity clustering'
    else:
        ylim = (0, 1.3)
        y_ticks = np.arange(0, 1.1, 0.5)
        xtick_labels = ['A', 'A sub', 'y', 'B', 'B sub']
        column_names = ['A', 'A sub', 'y',
                        'B', 'B sub']
        bar_colors = ['#0072B2', '#56B4E9', '#E69F00', '#009E73', '#B2DF8A']
        title = 'entropy'

    row_offsets = [0.1, 0.08, 0.04, 0.02, -0.015, -0.04, -0.06, -0.085]
    column_offsets = [-0.06, -0.015, 0.03, 0.075, -0.06, -0.015, 0.03, 0.075]
    row_index = i*2
    for j in range(8):
        if j == 4:
            row_index += 1
        data = model_dfs_list[j][i][column_names]
        axs[row_index, j % 4].boxplot(data.values, flierprops=flierprops, medianprops=medianprops)
        axs[row_index, j % 4].set_title(f'{model_list[j]}', fontsize=10)
        axs[row_index, j % 4].set_xticks(range(1, len(xtick_labels) + 1))
        axs[row_index, j % 4].set_xticklabels(xtick_labels, rotation=90)
        if ylim is not None:
            axs[row_index, j % 4].set_ylim(ylim[0] - 0.3, ylim[1])
            axs[row_index, j % 4].set_yticks(y_ticks)
        if j % 4 == 0:
            axs[row_index, j % 4].set_ylabel(y_axis_label, fontsize=9)
        axs[row_index, j % 4].tick_params(axis='both', which='major', labelsize=8)
        means = [np.mean(d) for d in data.values.T]
        y_min, y_max = axs[row_index, j % 4].get_ylim()
        annot_offset = (y_max - y_min) * 0.15
        for k, mean in enumerate(means, start=1):
            if not np.isnan(mean):
                axs[row_index, j % 4].text(k, y_max - annot_offset, f'{mean:.2f}', horizontalalignment='center', color='black', fontsize=6)

        if j < 4:
            adjust_subplot_position(axs[row_index, j % 4], row_offsets[i * 2], column_offsets[j], 0.75)
        else:
            adjust_subplot_position(axs[row_index, j % 4], row_offsets[i * 2 + 1], column_offsets[j], 0.75)

box_fig.text(0.5, 0.98, 'Input Similarity Clustering', ha='center', va='center', fontsize=13, fontweight='bold')
box_fig.text(0.5, 0.66, 'Hidden Similarity Clustering', ha='center', va='center', fontsize=13, fontweight='bold')
box_fig.text(0.5, 0.34, 'Output Similarity Clustering', ha='center', va='center', fontsize=13, fontweight='bold')
plt.show()

# def add_jitter(arr, scale=0.01):
#     return arr + np.random.uniform(-scale, scale, arr.shape)
#
# scatter_fig, scatter_axs = plt.subplots(4, 4, figsize=(fig_width, fig_height))
# x_ticks = np.arange(0, 1.2, 0.5)
# y_ticks = np.arange(0, 1.2, 0.2)
# for i, df in enumerate(model_dfs_list):
#     if i < 4:
#         row1, col1 = i//4, i % 4
#     else:
#         row1, col1 = i//4 + 1, i % 4
#     row2, col2 = row1 + 1, col1
#     y1_jitter = add_jitter(df[0]['B_alt_subgroup_correct'])
#     x1_jitter = add_jitter(df[1]['A sub'], 0.01)
#     scatter_axs[row1, col1].scatter(x1_jitter, y1_jitter, c='grey', s=20)
#     scatter_axs[row1, col1].set_title(f'{model_list[i]}')
#     scatter_axs[row1, col1].set_xlabel('A Sub Clustering Accuracy')
#     scatter_axs[row1, col1].set_ylabel('B Sub Predicting Accuracy')
#     scatter_axs[row1, col1].set_xlim(-0.1, 1.2)
#     scatter_axs[row1, col1].set_ylim(0, 1.1)
#     scatter_axs[row1, col1].set_xticks(x_ticks)
#     scatter_axs[row1, col1].set_yticks(y_ticks)
#
#     y2_jitter = add_jitter(df[0]['B_alt_subgroup_correct'])
#     x2_jitter = add_jitter(df[1]['B sub'], 0.01)
#
#     scatter_axs[row2, col2].scatter(x2_jitter, y2_jitter, c='grey', s=20)
#     scatter_axs[row2, col2].set_title(f'{model_list[i]}')
#     scatter_axs[row2, col2].set_xlabel('B Sub Clustering Accuracy')
#     scatter_axs[row2, col2].set_ylabel('B Sub Predicting Accuracy')
#     scatter_axs[row2, col2].set_xlim(-0.1, 1.2)
#     scatter_axs[row2, col2].set_ylim(0, 1.1)
#     scatter_axs[row2, col2].set_xticks(x_ticks)
#     scatter_axs[row2, col2].set_yticks(y_ticks)
#
# plt.tight_layout()
# plt.show()


109823
summary_file_path = '../../../AyB omit Results percentage.xlsx'
save_path = '../summary_plots'

w2v_df = pd.read_excel(summary_file_path, sheet_name='W2V_adam')
srn_df = pd.read_excel(summary_file_path, sheet_name='SRN_adam')
lstm_df = pd.read_excel(summary_file_path, sheet_name='LSTM_adam')
tf_df = pd.read_excel(summary_file_path, sheet_name='Transformer_adam')
accuracy_df = pd.read_excel(summary_file_path, sheet_name='Summary', nrows=16)
input_df = pd.read_excel(summary_file_path, sheet_name='Summary', skiprows=17, nrows=16)
output_df = pd.read_excel(summary_file_path, sheet_name='Summary', skiprows=34, nrows=16)
models = list(accuracy_df['Name'])
models_type = list((accuracy_df['Model Type']))
models_size = list((accuracy_df['Size']))

# Define the figure size (width x height in inches)
fig_width = 8.5
fig_height = 10

# Create the main figure with specified size
bar_fig = plt.figure(figsize=(fig_width, fig_height))
box_fig = plt.figure(figsize=(fig_width, fig_height))
title_list = ['Accuracy', 'Input Similarity Clustering', 'Output Similarity Clustering']


def adjust_subplot_position(ax, row_offset, col_offset, height_factor):
    pos = ax.get_position()
    new_pos = [pos.x0 + col_offset, pos.y0 + row_offset, pos.width, pos.height * height_factor]
    ax.set_position(new_pos)

# Loop to create three vertical subplots
for i in range(3):
    if i > 0:
        ylim = (0, 1.3)
        y_ticks = np.arange(0, 1.1, 0.5)
        # ylim = (0, 7.5)
        # y_ticks = np.arange(0, 7.5, 2)
        y_axis_label = 'Cluster Size'
        xtick_labels = ['A', 'A sub', 'y', 'B', 'B sub']
        column_names = ['A subcategories grouped', 'num As in subcategories', 'ys grouped',
                        'B subcategories grouped', 'num Bs in subcategories']
        bar_colors = ['#0072B2', '#56B4E9', '#E69F00', '#009E73', '#B2DF8A']
        if i == 1:
            rep_type = 'input'
            plot_df = input_df
            title = 'input similarity clustering'
        else:
            rep_type = 'output'
            plot_df = output_df
            title = 'output similarity clustering'
        column_names = [col_name + f'({rep_type})' for col_name in column_names]

    else:
        column_names = ['A_subgroup_correct', 'A_omit_correct',
                        'B_subgroup_correct', 'B_omit_correct']
        bar_colors = ['#0072B2', '#56B4E9', '#E69F00', '#009E73']
        plot_df = accuracy_df
        title = 'accuracy'
        y_axis_label = 'Accuracy'
        ylim = (0, 1.3)
        y_ticks = np.arange(0, 1.1, 0.5)
        xtick_labels = ['A sub', 'A omit', 'B sub', 'B omit']
    # Create a 2x4 grid of subplots inside the current row subplot
    row_offsets = [0.095, 0.075, 0.025, 0.005, -0.045, -0.065]
    column_offsets = [-0.06, -0.015, 0.03, 0.075, -0.06, -0.015, 0.03, 0.075]
    for j in range(8):
        ax = bar_fig.add_subplot(6, 4, i * 8 + j + 1)
        average_plot_df = plot_df[plot_df['Run'] == 'Average']
        std_plot_df = plot_df[plot_df['Run'] == 'STD'].reset_index()
        row_data = average_plot_df.loc[j, column_names]
        std_data = std_plot_df.loc[j, column_names]
        bars = ax.bar(row_data.index, row_data.values, yerr=std_data, color='white', edgecolor='black', capsize=3)  # Example plot
        ax.set_title(f'{models[j]}', fontsize=10)
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=90)
        ax.set_ylim(ylim[0], ylim[1])
        ax.tick_params(axis='both', which='major', labelsize=8)
        for bar in bars:
            yval = round(bar.get_height(), 2)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval / 2 + 0.15,
                f'{yval}',
                ha='center',
                va='center',
                fontsize=6,
                fontweight='bold'
            )
        if j < 4:
            adjust_subplot_position(ax, row_offsets[i*2], column_offsets[j], 0.75)
        else:
            adjust_subplot_position(ax, row_offsets[i*2+1], column_offsets[j], 0.75)
    flierprops = dict(marker='o', markersize=3, linestyle='none', markeredgecolor='black')
    medianprops = dict(color='black')
    for j in range(8):
        ax = box_fig.add_subplot(6, 4, i * 8 + j + 1)
        if models_type[j] == 'W2V':
            plot_df = w2v_df
            if i == 0:
                plot_df.loc[:, ['A_group_correct', 'y_group_correct', 'B_group_correct']] = np.nan
        elif models_type[j] == 'SRN':
            plot_df = srn_df
        elif models_type[j] == 'LSTM':
            plot_df = lstm_df
        else:
            plot_df = tf_df
        filtered_plot_df = plot_df[(plot_df['Size'] == models_size[j]) & (plot_df['Run'] != 'Summary')]
        row_data = filtered_plot_df[column_names]
        ax.boxplot(row_data.values, flierprops=flierprops, medianprops=medianprops)  # Example plot
        ax.set_title(f'{models[j]}', fontsize=10)
        ax.set_xticks(range(1, len(xtick_labels) + 1))
        ax.set_xticklabels(xtick_labels, rotation=90)
        ax.set_ylim(ylim[0]-0.3, ylim[1])
        ax.set_yticks(y_ticks)
        if j % 4 == 0:
            ax.set_ylabel(y_axis_label, fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        means = [np.mean(d) for d in row_data.values.T]
        y_min, y_max = ax.get_ylim()
        annot_offset = (y_max - y_min)*0.15
        for k, mean in enumerate(means, start=1):
            if not np.isnan(mean):
                ax.text(k, y_max - annot_offset, f'{mean:.2f}', horizontalalignment='center', color='black', fontsize=6)

        if j < 4:
            adjust_subplot_position(ax, row_offsets[i*2], column_offsets[j], 0.75)
        else:
            adjust_subplot_position(ax, row_offsets[i*2+1], column_offsets[j], 0.75)
        # for k in range(len(row_data.values)):
        #     mean_value = np.mean(row_data.values[k])
        #     ax.text(k + 1, mean_value + 0.1, f'{round(mean_value, 2):.2f}',
        #             ha='center', va='bottom', color='black', fontsize=4)


bar_fig.text(0.5, 0.98, 'Next Word Prediction Accuracy', ha='center', va='center', fontsize=13, fontweight='bold')
bar_fig.text(0.5, 0.65, 'Input Similarity Clustering', ha='center', va='center', fontsize=13, fontweight='bold')
bar_fig.text(0.5, 0.32, 'Output Similarity Clustering', ha='center', va='center', fontsize=13, fontweight='bold')
box_fig.text(0.5, 0.98, 'Next Word Prediction Accuracy', ha='center', va='center', fontsize=13, fontweight='bold')
box_fig.text(0.5, 0.65, 'Input Similarity Clustering', ha='center', va='center', fontsize=13, fontweight='bold')
box_fig.text(0.5, 0.32, 'Output Similarity Clustering', ha='center', va='center', fontsize=13, fontweight='bold')

# bar_fig.tight_layout(pad=2.0)
bar_fig.savefig('../summary_plots/omit_summary_bar_plot.png')
plt.close(bar_fig)

box_fig.savefig('../summary_plots/omit_summary_box_plot_adam.png')
plt.close(box_fig)


def add_jitter(arr, scale=0.01):
    return arr + np.random.uniform(-scale, scale, arr.shape)


srn_4_df = srn_df[(srn_df['Size'] == 4) & (srn_df['Run'] != 'Summary')]
srn_16_df = srn_df[(srn_df['Size'] == 16) & (srn_df['Run'] != 'Summary')]
lstm_4_df = lstm_df[(lstm_df['Size'] == 4) & (lstm_df['Run'] != 'Summary')]
lstm_16_df = lstm_df[(lstm_df['Size'] == 16) & (lstm_df['Run'] != 'Summary')]
w2v_4_df = w2v_df[(w2v_df['Size'] == 4) & (w2v_df['Run'] != 'Summary')]
w2v_16_df = w2v_df[(w2v_df['Size'] == 16) & (w2v_df['Run'] != 'Summary')]
tf_4_df = tf_df[(tf_df['Size'] == '4/2/4/4') & (tf_df['Run'] != 'Summary')]
tf_16_df = tf_df[(tf_df['Size'] == '16/4/16/4') & (tf_df['Run'] != 'Summary')]

dfs = [w2v_4_df, srn_4_df, lstm_4_df, tf_4_df, w2v_16_df, srn_16_df, lstm_16_df, tf_16_df]
df_names = ['W2V_4', 'SRN_4', 'LSTM_4', 'TF_4/2/4/4', 'W2V_16', 'SRN_16', 'LSTM_16', 'TF_16/4/16/4']
scatter_fig, scatter_axs = plt.subplots(4, 4, figsize=(fig_width, fig_height))
x_ticks = np.arange(0, 1.2, 0.5)
y_ticks = np.arange(0, 1.2, 0.2)
for i, df in enumerate(dfs):
    if i < 4:
        row1, col1 = i//4, i % 4
    else:
        row1, col1 = i//4 + 1, i % 4
    row2, col2 = row1 + 1, col1
    y1_jitter = add_jitter(df['B_subgroup_correct'])
    x1_jitter = add_jitter(df['num As in subcategories(input)'], 0.01)
    scatter_axs[row1, col1].scatter(x1_jitter, y1_jitter, c='grey', s=20)
    scatter_axs[row1, col1].set_title(f'{df_names[i]}')
    scatter_axs[row1, col1].set_xlabel('A Sub Clustering Accuracy')
    scatter_axs[row1, col1].set_ylabel('B Sub Predicting Accuracy')
    scatter_axs[row1, col1].set_xlim(-0.1, 1.2)
    scatter_axs[row1, col1].set_ylim(0, 1.1)
    scatter_axs[row1, col1].set_xticks(x_ticks)
    scatter_axs[row1, col1].set_yticks(y_ticks)

    y2_jitter = add_jitter(df['B_subgroup_correct'])
    x2_jitter = add_jitter(df['num Bs in subcategories(output)'], 0.01)

    scatter_axs[row2, col2].scatter(x2_jitter, y2_jitter, c='grey', s=20)
    scatter_axs[row2, col2].set_title(f'{df_names[i]}')
    scatter_axs[row2, col2].set_xlabel('B Sub Clustering Accuracy')
    scatter_axs[row2, col2].set_ylabel('B Sub Predicting Accuracy')
    scatter_axs[row2, col2].set_xlim(-0.1, 1.2)
    scatter_axs[row2, col2].set_ylim(0, 1.1)
    scatter_axs[row2, col2].set_xticks(x_ticks)
    scatter_axs[row2, col2].set_yticks(y_ticks)

plt.tight_layout()
plt.show()
