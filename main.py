import random
import pandas as pd

from dancing_shape import ShapeWorld

def generate_dataset(structures, actions, max_num_actions, changes, max_num_changes, shapes, max_num_shapes):
    columns = ['structure', 'actions', 'changes', 'shapes', 'num_var']
    data = []
    structure_to_num = {
            'direct': 2,
            'mediation': 3,
            'collision': 3, 
            'confounder': 3}
    
    for structure in structures:
        for i in range(max_num_actions):
            for j in range(max_num_changes):
                actions = random.sample(actions, i)
                changes = random.sample(changes, j)
                if structure != 'random':
                    num_var = structure_to_num[structure]
                    shapes = random.sample(num_var)
                    data.append([structure, actions, changes, shapes, num_var])
                else:
                    for num_var in range(3, max_num_shapes):
                        shapes = random.sample(shapes, num_var)
                        data.append([structure, actions, changes, shapes, num_var])
    
    df = pd.DataFrame(data=data, columns=columns)
    return df
