import itertools
import networkx as nx

import random
import json
import re

from prompt_templates import TEMPLATES
from utils import save_checkpoint, load_checkpoint, purge_checkpoint

class CausalWorld:

    def __init__(self, causal_structure, causal_flag, num_var=None, p_connect=0.5) -> None:
        """
        causal_structure: specified structure for hardcoded graphs or random for a random graph
        causal_flag: flags whether there is a causal relation between 0 and -1 in collision/confounder graphs
        num_var: number of variables, requried only for random graph
        """
        self.causal_structure = causal_structure
        self.causal_graph = nx.DiGraph()
        self.causal_flag = causal_flag
        self.p_connect = p_connect

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
            self.causal_graph = self.random_connected_graph()

    def _random_dag(self):
        length = self.num_var * (self.num_var - 1) // 2
        return [1 if random.random() < self.p_connect else 0 for _ in range(length)]
    
    def _dag_index(self, i, j):
        return self.num_var * i + j - (i + 1) * (i + 2) // 2
    
    def _is_connected(self, dag):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_var))
        
        for i in range(self.num_var):
            for j in range(i + 1, self.num_var):
                idx = self._dag_index(i, j)
                if dag[idx] == 1:
                    G.add_edge(i, j)
        
        return len(nx.descendants(G, 0)) + 1 == self.num_var
    
    def _complement(self, dag):
        return [0 if x else 1 for x in dag]
    
    def random_connected_graph(self):
        dag = self._random_dag()
        dag = dag if self._is_connected(dag) else self._complement(dag)
        
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_var))
        
        for i in range(self.num_var):
            for j in range(i + 1, self.num_var):
                idx = self._dag_index(i, j)
                if dag[idx] == 1:
                    G.add_edge(i, j)
        
        return G
        
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
    def __init__(self, causal_structure, causal_flag, prompt_template, 
                 model_name, pipeline_handler,
                 actions=None, changes=None, shapes=None, num_var=None,
                 experiment_ver=None) -> None:
        super().__init__(causal_structure, causal_flag, num_var)
        self.generate_variables(actions, changes, shapes)
        self.template = prompt_template
        self.prompt_templates = TEMPLATES[prompt_template]
        print(self.prompt_templates)
        self.pipeline_handler = pipeline_handler
        self.active_shapes = []
        self.result = None
        self.error_mode = None
        self.model_name = model_name
        self.checkpoint_path = "./checkpoints/{}_{}_shape_world.pkl".format(model_name, prompt_template)
        self.experiment_ver = experiment_ver


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
        
        if self.model_name == "human":
            prompt = '\n'.join([first_section_prompt, json_prompt])
        else:
            if state == 'initial':
                if 'deepseek' in self.model_name or 'mistral' in self.model_name:
                    prompt = [{"role": "user", "content": '\n'.join([first_section_prompt, json_prompt])}]
                else:
                    prompt = [
                        {"role": "system", "content": first_section_prompt}, 
                        {"role": "user", "content": json_prompt}]
                
            else:
                prompt = {"role": "user", "content": '\n'.join([first_section_prompt, json_prompt])}
  
        return prompt
    

    def collect_response(self, prompt):
        if type(prompt) is list:
            self.chat = prompt
        else:
            self.chat.append(prompt)
        
        response, self.chat = self.pipeline_handler.collect_response(self.chat)
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
        self.question = "Does {} moving cause {} to move?".format(cause, effect)
        self.answer = self.check_causal_path(self.shapes.index(cause), self.shapes.index(effect))


        curr_state = "initial"
        initial_prompt = self.generate_prompt(curr_state)
        response = self.collect_response(initial_prompt)
        step = 0
        max_step = len(self.shapes) * len(self.actions)

        while response and curr_state != 'answer' and step < max_step:
            curr_state, prompt = self.interaction_step(curr_state, response)
            if curr_state is None:
                response = None
                break
            if curr_state == "interaction":
                step += 1
            response = self.collect_response(prompt)

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
        if self.experiment_ver == 'comprehensive':
            nodes = list(self.causal_graph.nodes())
            node_and_descendants = [
                [n] + list(nx.descendants(self.causal_graph, n)) for n in nodes]
                    
            setups = []
            for i in range(len(node_and_descendants)):
                setups.extend(list(itertools.combinations(node_and_descendants, i)))
            
            for i in range(len(setups)):
                setups[i] = frozenset(itertools.chain(*setups[i]))
        elif self.experiment_ver == 'hard':
            setups = [frozenset(range(len(self.shapes)))]
        else:
            setups = []
        
        return set(setups)
    
    def _run_experiment_comprehensive(self, setup, processed_setups, result_table, model_path, error_log_path):
        
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

                if not result:
                    log_entry = {
                        "error": self.error_mode,
                        "cause": cause,
                        "effect": effect,
                        "setup": list(setup),
                        "conversational_history": self.chat if hasattr(self, 'chat') else []
                    }
                    with open(error_log_path, "a") as log:
                        log.write(json.dumps(log_entry) + "\n")

                processed_setups.add((frozenset(setup), cause, effect))

        
    def _run_experiment_hard(self, setup, processed_setups, result_table, model_path, error_log_path):
        sample = random.sample(list(itertools.combinations(self.shapes, 2)), 6)
        for (cause, effect) in sample:
            if (frozenset(setup), cause, effect) in processed_setups:
                continue
            self.error_mode = None
            
            error, result, step = self.interaction_loop(list(setup), cause, effect, model_path)
            entries = {
                'structure': self.causal_structure,
                'setup': setup,
                'cause': cause,
                'effect': effect,
                'ground_truth': self.answer,
                'error': error,
                'result': result,
                'n_step': step,
            }

            for key, value in entries.items():
                result_table[key].append(value)

            if not result:
                log_entry = {
                    "error": self.error_mode,
                    "cause": cause,
                    "effect": effect,
                    "setup": list(setup),
                    "conversational_history": self.chat if hasattr(self, 'chat') else []
                }
                with open(error_log_path, "a") as log:
                    log.write(json.dumps(log_entry) + "\n")

            processed_setups.add((frozenset(setup), cause, effect))
                    

    def run_experiment(self, model_path=None):

        checkpoint = load_checkpoint(self.checkpoint_path)
        error_log_path = "./failure_cases/{}_{}_{}.jsonl".format(self.model_name, self.causal_structure, self.template)
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
            if self.experiment_ver == "comprehensive":
                self._run_experiment_comprehensive(setup, processed_setups, result_table, model_path, error_log_path)
            elif self.experiment_ver == "hard":
                self._run_experiment_hard(setup, processed_setups, result_table, model_path, error_log_path)

            save_checkpoint(self.checkpoint_path, {
                'result_table': result_table,
                'processed_setups': processed_setups,
                'setups': setups
            })
        
        purge_checkpoint(self.checkpoint_path)
        
        return result_table