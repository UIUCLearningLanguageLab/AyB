import os
from pathlib import Path
from distributional_models.scripts.load_model import load_model, load_models
from distributional_models.tasks.states import States
from ayb.src.evaluate import evaluate_model

run_path = Path('../../runs/')
# run_path = Path('../AdamW_no_omit_runs/runs/')
selected_param = 'param_001'
epoch_evaluated = [f'e{epoch}' for epoch in range(0, 2001, 100)]
params, corpus, model_dict = load_models(run_path, selected_param, epoch_selected=[20], model_selected=[22])
save_path = os.path.join(run_path, params['save_path'][9:], 'extra_eval')
os.makedirs(save_path, exist_ok=True)

for model_index, model_list in model_dict.items():
    for i, model in enumerate(model_list):
        states_evaluation = States(model=model, corpus=corpus, params=params, save_path=save_path, layer='hidden')
        states_evaluation.get_vocab_category_subcategory_into()
        states_evaluation.generate_contrast_pairs()
        states_evaluation.get_weights()
        states_evaluation.export_weights_to_csv()
        states_evaluation.generate_input_sequences()
        fig_dict = states_evaluation.evaluate_states(position=-1)
        for input_token, fig in fig_dict.items():
            fig.savefig(save_path+f'/Diagnose_{model_index}_{model.epoch-1}_{input_token}.png')
        print('evaluating states')