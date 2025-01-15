import os
import random
import itertools
import argparse

import pandas as pd
from tqdm import tqdm

from dancing_shape import ShapeWorld

def generate_dataset(structures, actions, changes, shapes):
    """
    Returns a dataframe containing dataset parameters. 

    structure: causal structure
    actions: valid actions
    max_num_actions: maximum number of possible actions
    ...
    """
    columns = ['structure', 'actions', 'changes', 'shapes', 'num_var']
    data = []
    structure_to_num = {
            'direct': 2,
            'mediation': 3,
            'confounder': 3}
    
    for structure in structures:
        for (i, j) in itertools.product(range(len(actions)), range(len(changes))):
            actions = random.sample(actions, i)
            changes = random.sample(changes, j)
            if structure != 'random':
                num_var = structure_to_num[structure]
                shapes = random.sample(num_var)
                data.append([structure, actions, changes, shapes, num_var])
            else:
                for num_var in range(3, len(shapes)):
                    shapes = random.sample(shapes, num_var)
                    data.append([structure, actions, changes, shapes, num_var])
    
    df = pd.DataFrame(data=data, columns=columns)
    return df


def run_experiments(dataset_df, num_rep, model, model_path, prompt_template_name):
    dataset = dataset_df.values.tolist()
    model_results = []
    for row in tqdm(dataset, desc="Progress by Group"):
        causal_structure, actions, changes, shapes, num_var = tuple(row)
        model_result_row = [causal_structure, len(actions), len(changes), num_var]
        for _ in range(num_rep):
            temp_results = []
            ground_truths = []
            s_world = ShapeWorld(causal_structure, prompt_template_name, model, actions, changes, shapes, num_var)

            for (var_1, var_2) in itertools.combinations(s_world.shape_changes):
                for (cause, effect) in [(var_1, var_2), (var_2, var_1)]:
                    ground_truth = s_world.check_causal_path(cause, effect)
                    result = s_world.interaction_loop(cause, effect, model_path)
                    ground_truths.append(ground_truth)
                    temp_results.append(result)
            model_result_row.append(ground_truths)
            model_result_row.append(temp_results)

        model_results.append(model_result_row)
    
    result_df = pd.DataFrame(data=model_results, columns=['structure', 'num_actions', 
                                                          'num_changes', 'num_shapes',
                                                          'ground_truths', 'results'])
    
    return result_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default='./data/test_df.csv')
    parser.add_argument('result_path', type=str, default='./results/test_result.csv')
    parser.add_argument('model', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('num_rep', type=int, default=1)
    parser.add_argument('prompt_template', type=str, default='basic_limit_steps')

    args = parser.parse_args()

    if os.path.exists(args.data_path):
        data_df = pd.read_csv(args.data_path, index_col=0)
    else:
        actions = ['touch']
        changes = ['moving']
        shapes = ["circle", "square", 'triangle']
        structures = ['direct', 'mediation', 'confounder', 'random']

        data_df = generate_dataset(structures, actions, changes, shapes)
        data_df.to_csv(args.data_path, index=True)

    result_df = run_experiments(data_df, args.num_rep, args.model, args.model_path, args.prompt_template)
    result_df.to_csv(args.result_path, index=True)
