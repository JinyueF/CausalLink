import os
import random
import argparse

import pandas as pd
from ast import literal_eval

from dancing_shape import ShapeWorld
from utils import save_checkpoint, load_checkpoint, purge_checkpoint
from pipeline_handler import PipelineHandler

def generate_dataset(num_rep, structures, all_shapes, min_num_var=4, max_num_var=6):
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
            for num_var in range(min_num_var, max_num_var):
                for _ in range(num_rep):
                    shapes = random.sample(all_shapes, num_var)
                    data.append([structure, True, shapes, num_var])
    
    df = pd.DataFrame(data=data, columns=columns)
    return df


def run_experiments(dataset_df, source, model, model_path, prompt_template_name, checkpoint_path):
    dataset = dataset_df.values.tolist()
    result_table = {}

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        result_table = checkpoint['result_table']
        row_num = checkpoint['row_num']
    else:
        result_table = {}
        row_num = 0

    while row_num < dataset.shape[0]:
        causal_structure, causal_flag, shapes, num_var = tuple(dataset.iloc[row_num, :])
        shapes = literal_eval(shapes) if type(shapes) is str else shapes
        api_key = os.environ[args.api_key] if args.api_key else ''
        pipeline_handler = PipelineHandler(source, model_path, api_key, args.temperature)
        s_world = ShapeWorld(
            causal_structure, causal_flag, prompt_template_name, model,
            pipeline_handler, shapes=shapes, num_var=num_var)
        curr_result = s_world.run_experiment(model_path)
        if result_table == {}:
            result_table = curr_result
        else:
            result_table = {key: result_table[key] + curr_result[key] for key in result_table}
        
        save_checkpoint(checkpoint_path, {
                        'result_table': result_table,
                        'row_num': row_num
                    })
        row_num += 1
    
    result_df = pd.DataFrame(result_table)
    purge_checkpoint(checkpoint_path)
    
    return result_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path to the csv file containing the dataset.")
    parser.add_argument('--result_path', type=str, help="Path to the csv file containing the results.")
    parser.add_argument('--source', type=str,
                        help="The source of the model being tested.",
                        choices=['huggingface', 'google', 'deepseek', 'vec-inf'])
    parser.add_argument('--model', type=str,
                        help="Model name. Must be the correct model aliases if using API calls.")
    parser.add_argument('--model_path', type=str,
                        help="Model paths for huggingface models or base urls for API calls.")
    parser.add_argument('--api_key', type=str, help="Name of environmental variable storing the API key.")
    parser.add_argument('--temperature', type=float, help="Generation temperature for the model.")
    parser.add_argument('--num_rep', type=int, default=1, help="Number of repitition of experiments. Default to 1.")
    parser.add_argument('--prompt_template', type=str,
                        help="Name of the prompting template. Must be one of the key names in prompting_template.py.")

    args = parser.parse_args()
    checkpoint_path = "./checkpoints/{}_{}_experiment_checkpoint.pkl".format(args.model, args.prompt_template)

    if os.path.exists(args.data_path):
        data_df = pd.read_csv(args.data_path, index_col=0)
    else:
        shapes = ["circle", "square", "triangle", "rectangle", "hexagon", "pentagon", "octagon", "ellipse"]
        structures = ['random']

        data_df = generate_dataset(args.num_rep, structures, shapes)
        data_df.to_csv(args.data_path, index=True)

    result_df = run_experiments(data_df, args.source, args.model, args.model_path, args.prompt_template, checkpoint_path)
    result_df.to_csv(args.result_path, index=True)
