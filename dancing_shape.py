import itertools
import networkx as nx
import torch
from transformers import pipeline


import random
import json
import re

from prompt_templates import TEMPLATES
from utils import save_checkpoint, load_checkpoint

class CausalWorld:

    def __init__(self, causal_structure, causal_flag, num_var=None) -> None:
        """
        causal_structure: specified structure for hardcoded graphs or random for a random graph
        causal_flag: flags whether there is a causal relation between 0 and -1 in collision/confounder graphs
        num_var: number of variables, requried only for random graph
        """
        self.causal_structure = causal_structure
        self.causal_graph = nx.DiGraph()
        self.causal_flag = causal_flag

        self.structure_to_num = {
            'direct': 2,
            'mediation': 3,
            'collider': 3, 
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
            self.causal_graph.add_edge(self.var_list[1], self.var_list[2])
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
    def __init__(self, causal_structure, causal_flag, prompt_template, model, actions=None, changes=None, shapes=None, num_var=None) -> None:
        super().__init__(causal_structure, causal_flag, num_var)
        self.generate_variables(actions, changes, shapes)
        self.prompt_templates = TEMPLATES[prompt_template]
        self.model = model
        self.active_shapes = []
        self.result = None
        self.error_mode = None
        self.checkpoint_path = "{}_shape_world.pkl".format(model)


    def generate_variables(self, actions, changes, shapes):
        """
        Stores an action dictionary recording changes to shapes after an action (intervention) and 
        a change dictiionary containing shapes and changes to be used with the causal graph. 
        """
        if not shapes:
            shapes = ["circle", "square", "triangle", "rectangle", "hexagon", "pentagon", \
                    "octagon", "ellipse"]
        if not actions:
            actions = ['move', 'hold']
        if not changes:
            changes = ['moving']
        
        self.actions = actions
        self.changes = changes

        self.shapes = random.sample(shapes, k=self.num_var)
        self.shape_changes = []
        for shape in self.shapes:
            self.shape_changes.append((shape, random.choice(changes)))
                
        
    def apply_intervention(self, shape, action):
        if shape not in self.shapes or action not in self.actions:
            self.error_mode = 'invalid action'
            return
        
        source_node = self.shapes.index((shape))
        if action == 'move':
            successors = nx.descendants(self.causal_graph, source_node)
            self.active_shapes.append(source_node)
            self.active_shapes.extend(list(successors))
            self.active_shapes = list(set(self.active_shapes))
        else:
            self.deactivate_node(source_node)


    def deactivate_node(self, node):
        def has_active_parent(node):
            parents = list(self.causal_graph.predecessors(node))
            return any(parent in self.active_shapes for parent in parents)
            
        if not has_active_parent(node) and node in self.active_shapes:
            self.active_shapes.remove(node)
        
            for child in self.causal_graph.successors(node):
                if not has_active_parent(child):
                    self.deactivate_node(child)
    

    def check_result(self, parsed_response):
        try:
            result = parsed_response['answer']
            if result in ['yes', 'no']:
                return (self.answer and result == 'yes') or (not self.answer and result == 'no')
            else:
                return None
        except KeyError:
            pass
        return None
        

    def format_current_changes(self):
        # Create a dictionary of shape statuses
        current_shape_changes = {
            s: 'moving' if idx in self.active_shapes else 'static'
            for idx, s in enumerate(self.shapes)
        }

        # Format the changes into a readable string
        return '\n'.join([f"{key} is {value}" for key, value in current_shape_changes.items()])


    def generate_prompt(self, state):
        
        if state == "initial":
            first_section_prompt = self.prompt_templates['system'] # system prompt
            json_prompt = self.prompt_templates['initial'].format(self.format_current_changes(), self.question, str(self.shapes), str(self.actions))

        elif state == "choice":
            first_section_prompt = "Following your last action, the current states of shapes are: " + self.format_current_changes()
            json_prompt = self.prompt_templates['choice'].format(self.question)
        
        elif state == 'interaction':
            first_section_prompt = ''
            json_prompt = self.prompt_templates['interaction'].format(str(self.shapes), str(self.actions))
        
        elif state == 'answer':
            first_section_prompt = ''
            json_prompt = self.prompt_templates['answer'].format(self.question)
        
        if self.model == "human":
            prompt = '\n'.join([first_section_prompt, json_prompt])
        elif self.model.startswith('hf'):
            if state == 'initial':
                if 'deepseek' in self.model or 'mistral' in self.model:
                    prompt = [{"role": "user", "content": '\n'.join([first_section_prompt, json_prompt])}]
                else:
                    prompt = [
                        {"role": "system", "content": first_section_prompt}, 
                        {"role": "user", "content": json_prompt}]
                
            else:
                prompt = {"role": "user", "content": '\n'.join([first_section_prompt, json_prompt])}
  
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
            raw_response = pipe(self.chat, max_new_tokens=2048)
            response = raw_response[0]['generated_text'][-1]['content']
            print(response)
            parsed = self.parse_intervention(response)

        return parsed
    
    def interaction_step(self, last_state, last_response):
        try:
            if last_state in ['initial', 'interaction'] \
                or (last_state == 'choice' and 'next' not in last_response):
                curr_state = 'choice'
                self.apply_intervention(last_response['shape'], last_response['action'])
            elif last_state == 'choice':
                if last_response['next'] == "continue interaction":
                    curr_state = 'interaction'
                elif last_response['next'] == 'answer the question':
                    curr_state = 'answer'
                else:
                    curr_state = 'choice'
            
            prompt = self.generate_prompt(curr_state)
            return curr_state, prompt
        except KeyError:
            return None, None
            

    def interaction_loop(self, initial_active_shapes, cause, effect, model_path=None):
        """
        Performs interaction for the question does cause cause effect?
        """
        self.active_shapes = initial_active_shapes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model
        self.question = "Does {} moving cause {} to move?".format(cause, effect)
        self.answer = self.check_causal_path(self.shapes.index(cause), self.shapes.index(effect))

        if model.startswith('hf'):
            pipe = pipeline("text-generation", model_path, torch_dtype=torch.bfloat16, device=device, temperature=0.6)
        elif model == 'human':
            pipe = None
        
        curr_state = "initial"
        initial_prompt = self.generate_prompt(curr_state)
        print(initial_prompt)
        response = self.collect_response(initial_prompt, pipe)
        step = 0
        max_step = len(self.shapes) * len(self.actions)

        while response and curr_state != 'answer' and step < max_step:
            curr_state, prompt = self.interaction_step(curr_state, response)

            print(prompt)
            if curr_state is None:
                response = None
                break
            if curr_state == "interaction":
                step += 1
            response = self.collect_response(prompt, pipe)

        if not response:
            self.error_mode = "invalid format"
            self.result = None
        elif step == max_step and not self.error_mode:
            self.error_mode = "too many attempts"
            self.result = None
        else:
            self.result = self.check_result(response)
            if self.result is None:
                self.error_mode = "invalid answer"
        
        return self.error_mode, self.result, step
    

    def get_initial_setups(self):
        nodes = list(self.causal_graph.nodes())
        node_and_descendants = [
            [n] + list(nx.descendants(self.causal_graph, n)) for n in nodes]
        
        setups = []
        for i in range(len(node_and_descendants)):
            setups.extend(list(itertools.combinations(node_and_descendants, i)))
        
        for i in range(len(setups)):
            setups[i] = frozenset(itertools.chain(*setups[i]))
        
        return set(setups)
        

    def run_experiment(self, model_path=None):

        checkpoint = load_checkpoint(self.checkpoint_path)
        if checkpoint:
            result_table = checkpoint['result_table']
            processed_setups = checkpoint['processed_setups']
            setups = checkpoint['setups']
        else:
            result_table = {'structure': [], 'setup':[], 'cause':[], 'effect':[], 'ground_truth':[],
                            'error':[], 'result':[], 'n_step':[]}
            processed_setups = set()
            setups = self.get_initial_setups()

        for setup in setups:
            for (var_1, var_2) in itertools.combinations(self.shapes, 2):
                for (cause, effect) in [(var_1, var_2), (var_2, var_1)]:
                    if (frozenset(setup), cause, effect) in processed_setups:
                        continue

                    self.error_mode = None
                    error, result, step = self.interaction_loop(list(setup), cause, effect, model_path)
                    result_table['structure'].append(self.causal_structure)
                    result_table['setup'].append(setup)
                    result_table['cause'].append(cause)
                    result_table['effect'].append(effect)
                    result_table['ground_truth'].append(self.answer)
                    result_table['error'].append(error)
                    result_table['result'].append(result)
                    result_table['n_step'].append(step)

                    processed_setups.add((frozenset(setup), cause, effect))
                
                    # Save checkpoint
                    save_checkpoint(self.checkpoint_path, {
                        'result_table': result_table,
                        'processed_setups': processed_setups,
                        'setups': setups
                    })

        return result_table