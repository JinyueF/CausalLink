import os
import random
import itertools
import argparse

import pandas as pd
from tqdm import tqdm
from ast import literal_eval

from dancing_shape import ShapeWorld

def generate_dataset(structures, all_shapes):
    """
    Returns a dataframe containing dataset parameters. 

    structure: causal structure
    actions: valid actions
    max_num_actions: maximum number of possible actions
    ...
    """
    columns = ['structure', 'causal_flag', 'shapes', 'num_var']
    data = []
    structure_to_num = {
            'direct': 2,
            'mediation': 3,
            'confounder': 3}
    
    for structure in structures:
            
        if structure != 'random':
            num_var = structure_to_num[structure]
            shapes = random.sample(all_shapes, num_var)
            data.append([structure, True, shapes, num_var])
            if structure == 'confounder':
                data.append([structure, False, shapes, num_var])
        
        else:
            for num_var in range(3, len(all_shapes)+1):
                
                shapes = random.sample(all_shapes, num_var)
                data.append([structure, True, shapes, num_var])
    
    df = pd.DataFrame(data=data, columns=columns)
    return df


def run_experiments(dataset_df, num_rep, model, model_path, prompt_template_name):
    dataset = dataset_df.values.tolist()

    for row in tqdm(dataset, desc="Progress by Group"):
        causal_structure, causal_flag, shapes, num_var = tuple(row)
        shapes = literal_eval(shapes) if type(shapes) is str else shapes
        result_table = {}
        for _ in range(num_rep):
            
            s_world = ShapeWorld(causal_structure, causal_flag, prompt_template_name, model, shapes=shapes, num_var=num_var)
            curr_result = s_world.run_experiment(model_path)
            if result_table == {}:
                result_table = curr_result
            else:
                result_table = {key: result_table[key] + curr_result[key] for key in result_table}
    
    result_df = pd.DataFrame(result_table)
    
    return result_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/test_df.csv')
    parser.add_argument('--result_path', type=str, default='./results/test_result.csv')
    parser.add_argument('--model', type=str, default='hf-qwen')
    parser.add_argument('--model_path', type=str, default='/model-weights/Qwen2.5-14B-Instruct')
    parser.add_argument('--num_rep', type=int, default=1)
    parser.add_argument('--prompt_template', type=str, default='basic')

    args = parser.parse_args()

    if os.path.exists(args.data_path):
        data_df = pd.read_csv(args.data_path, index_col=0)
    else:
        shapes = ["circle", "square", "triangle"]
        structures = ['direct', 'mediation', 'confounder', 'random']

        data_df = generate_dataset(structures, shapes)
        data_df.to_csv(args.data_path, index=True)

    result_df = run_experiments(data_df, args.num_rep, args.model, args.model_path, args.prompt_template)
    result_df.to_csv(args.result_path, index=True)
