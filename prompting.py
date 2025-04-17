from prompt_templates import TEMPLATES

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
    else:
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