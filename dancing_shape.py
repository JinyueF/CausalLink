import networkx as nx
import torch
from transformers import pipeline


import random
import json
import re

from prompt_templates import TEMPLATES

class CausalWorld:

    def __init__(self, causal_structure, causal_flag=True, num_var=None) -> None:
        """
        causal_structure: specified structure for hardcoded graphs or random for a random graph
        causal_flag: flags whether there is a causal relation between 0 and -1 in collision/confounder graphs
        num_var: number of variables, requried only for random graph
        """
        self.causal_structure = causal_structure
        self.causal_graph = nx.DiGraph()
        self.causal_flag = causal_flag # Flags whether there is a causal relation between the variables of interest 

        self.structure_to_num = {
            'direct': 2,
            'mediation': 3,
            'collision': 3, 
            'confounder': 3,
            'random': num_var
        }

        self.num_var = self.structure_to_num[self.causal_structure]
        if self.num_var:
            self.var_list = list(range(self.num_var))
        else:
            raise ValueError
        self.generate_causal_relations()
            

    def generate_causal_relations(self) -> None:

        if self.causal_structure == 'direct':
            # A --> B
            self.causal_graph.add_edge(self.var_list[0], self.var_list[1])
            self.cause_variable = self.var_list[0]
            self.effect_variable = self.var_list[1]
        elif self.causal_structure == 'mediation':
            # A --> B --> C
            self.causal_graph.add_edge(self.var_list[0], self.var_list[1])
            self.cause_graph.add_edge(self.var_list[1], self.var_list[2])
            self.cause_variable = self.var_list[0]
            self.effect_variable = self.var_list[2]
        elif self.causal_structure == 'confounder':
            # A <-- B --> C
            self.causal_graph.add_edge(self.var_list[1], self.var_list[0])
            self.causal_graph.add_edge(self.var_list[1], self.var_list[2])
            self.cause_variable = self.var_list[0]
            self.effect_variable = self.var_list[2]
            if self.causal_flag:
                self.causal_graph.add_edge(self.var_list[0], self.var_list[2])
        elif self.causal_structure == 'collider':
            # A --> B <-- C
            self.causal_graph.add_edge(self.var_list[0], self.var_list[1])
            self.causal_graph.add_edge(self.var_list[2], self.var_list[1])
            self.cause_variable = self.var_list[0]
            self.effect_variable = self.var_list[2]
            if self.causal_flag:
                self.causal_graph.add_edge(self.var_list[0], self.var_list[2])
        else:
            # randomly generate a causal graph
            g = nx.gnp_random_graph(self.num_var, 0.5, directed=True)
            self.causal_graph = nx.DiGraph([(u, v) for (u, v) in g.edges() if u < v])
        
    def check_causal_path(self, cause_var, effect_var) -> bool:
        return nx.has_path(self.causal_graph, cause_var, effect_var)
    
    def parse_intervention(self, text_input):
        json_pattern = r'\{[\s\S]*?\}'
        match = re.search(json_pattern, text_input)
        json_object = None
        try:
            json_object = json.loads(match.group())
        except:
            pass
        
        return json_object


            
class ShapeWorld(CausalWorld):
    def __init__(self, causal_structure, prompt_template, model, causal_flag=True, num_var=None) -> None:
        super().__init__(causal_structure, causal_flag, num_var)
        self.generate_variables()
        self.prompt_templates = TEMPLATES[prompt_template]
        self.model = model
        self.current_shape_changes = {}
    

    def generate_variables(self, actions=None, changes=None, shapes=None):
        """
        Stores an action dictionary recording changes to shapes after an action (intervention) and 
        a change dictiionary containing shapes and changes to be used with the causal graph. 
        """
        if not shapes:
            shapes = ["circle", "square", "triangle", "rectangle", "hexagon", "pentagon", \
                    "octagon", "ellipse"]
        if not actions:
            actions = ['touch', 'push']
        if not changes:
            changes = ['enlarging', 'moving']
        
        self.actions = actions
        self.changes = changes

        self.shapes = random.sample(shapes, k=self.num_var)
        self.shape_changes = []
        for shape in self.shapes:
            self.shape_changes.append((shape, random.choice(changes)))

        action_to_changes = {}
        for shape in self.shapes:
            shape_dict = {}
            for action in actions:
                shape_dict[action] = random.choice(changes+['not changing'])
            action_to_changes[shape] = shape_dict

        # Ensure shape changes with no incoming edges can be initiated by a valid action
        root_nodes = [node for node in self.causal_graph if self.causal_graph.in_degree(node) == 0]
        for i in root_nodes:
            shape, change = self.shape_changes[i]
            if change not in action_to_changes[shape].values():
                key = random.choice(actions)
                action_to_changes[shape][key] = change
        
        self.action_to_changes = action_to_changes

        
    def apply_intervention(self, shape, action):
        change = self.action_to_changes[shape][action]
        if (shape, change) in self.shape_changes:
            source_node = self.shape_changes.index((shape, change))
            successors = nx.dfs_successors(self.causal_graph, source_node)
            result = [self.shape_changes[i] for i in set(list(successors.keys()) + sum(list(successors.values()), []))]
        else:
            result = [(shape, change)]
        
        self.current_shape_changes[' '.join([action, shape])] = [item[0]+' '+item[1] for item in result]
    

    def check_result(self, parsed_response):
        result = parsed_response['answer']
        if result in ['yes', 'no']:
            return (self.answer and result == 'yes') or (not self.answer and result == 'no')
        else:
            return None
        

    def format_current_changes(self):
        return '\n'.join(
            [": ".join([key, ', '.join(value)]) for key, value in self.current_shape_changes.items()]
        )


    def generate_prompt(self, state, parsed_response):

        question_prompt = self.prompt_templates['question'].format(
                                    self.question, str(self.shapes), str(self.actions))
        
        if state == "initial":
            first_section_prompt = self.prompt_templates['system'] # system prompt
            json_prompt = self.prompt_templates['initial']

        elif state == "choice":
            first_section_prompt = self.format_current_changes()
            json_prompt = self.prompt_templates['choice']
        
        elif state == 'interaction':
            first_section_prompt = ''
            json_prompt = self.prompt_templates['interaction']
        
        elif state == 'answer':
            first_section_prompt = self.format_current_changes()
            json_prompt = self.prompt_templates['answer']
            
        
        if self.model == "human":
            prompt = '\n'.join([first_section_prompt, question_prompt, json_prompt])
        elif self.model.startswith('hf'):
            if state == 'initial':
                prompt = [
                    {"role": "system", "content": first_section_prompt}, 
                    {"role": "user", "content": '\n'.join([question_prompt, json_prompt])}]
            else:
                prompt = {"role": "user", "content": '\n'.join([first_section_prompt, question_prompt, json_prompt])}
  
        return prompt
    

    def collect_response(self, prompt, pipe):

        if self.model == 'human':
            response = input(prompt).lower()
            parsed = self.parse_intervention(response)
            step = 0
            while not parsed and step < 3:
                response = input("Invalid input, please try again").lower()
                parsed = self.parse_intervention(response)
                step += 1
        elif self.model.startswith('hf'):
            if type(prompt) is list:
                self.chat = prompt
            else:
                self.chat.append(prompt)
            raw_response = pipe(self.chat, max_new_tokens=512)
            response = raw_response[0]['generated_text'][-1]['content']
            print(response)
            parsed = self.parse_intervention(response)

        return parsed
    
    def interaction_step(self, last_state, last_response):
        if last_state in ['initial', 'interaction']:
            curr_state = 'choice'
            self.apply_intervention(last_response['shape'], last_response['action'])
            
        elif last_state == 'choice':
            if last_response['next'] == "continue interaction":
                curr_state = 'interaction'
            elif last_response['next'] == 'answer the question':
                curr_state = 'answer'
        
        prompt = self.generate_prompt(curr_state, last_response)
        return curr_state, prompt
            

    def interaction_loop(self, cause, effect, model_path=None):
        """
        Performs interaction for the question does cause cause effect?
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model

        cause_text = ' '.join(cause)
        effect_text = ' '.join(effect)
        self.question = "Does {} cause {}?".format(cause_text, effect_text)
        self.answer = self.check_causal_path(self.shape_changes.index(cause), self.shape_changes.index(effect))

        if model.startswith('hf'):
            pipe = pipeline("text-generation", model_path, torch_dtype=torch.bfloat16, device=device)
        elif model == 'human':
            pipe = None
        
        curr_state = "initial"
        initial_prompt = self.generate_prompt(curr_state, None)
        response = self.collect_response(initial_prompt, pipe)
        curr_state, prompt = self.interaction_step(curr_state, response)
        step = 1
        max_step = len(self.shapes) * len(self.actions)

        while response and curr_state != 'answer' and step < max_step:
            response = self.collect_response(prompt, pipe)
            curr_state, prompt = self.interaction_step(curr_state, response)
            step += 1

        if curr_state == 'answer':
            result = self.check_result(self.collect_response(prompt, pipe))
        else:
            return None
        
        return result


if __name__ == '__main__':
    s_world = ShapeWorld('direct', 'basic', 'human', True, 5)
    print(nx.to_dict_of_dicts(s_world.causal_graph))
    result = s_world.interaction_loop(s_world.shape_changes[0], s_world.shape_changes[1], '/model-weights/Llama-3.2-1B-Instruct')
    print(result)