import torch

from transformers import pipeline
from openai import OpenAI
from google import genai

class PipelineHandler:

    def __init__(self, source, model, model_path, api_key, temperature):

        self.source = source
        self.model = model
        self.model_path = model_path
        self.temperature = temperature

        self.model_pipeline_dict = {
            'huggingface':{
                'func': pipeline,
                'args':{
                    'task': 'text-generation', 'model': model_path, 'torch_dytype': torch.float16, 'device_map': 'auto'
                }
            },
            'openai':{
                'func': OpenAI,
                'args': {
                    'api_key': api_key
                }
            },
            'deepseek':{
                'func': OpenAI,
                'args': {
                    'api_key': api_key,
                    'base_url': model_path
                }
            },
            'google':{
                'func': genai.Client,
            },
            'vec-inf':{
                'func': OpenAI,
                'args': {
                    'api_key': 'EMPTY',
                    'base_url': model_path
                }
            }
        }

        if source == 'google':
            f = self.model_pipeline_dict[source]['func']
            client = f(api_key=api_key, http_options={'api_version':'v1alpha'})
            self.pipe = client.chats.create(model=self.model)
        else:
            f = self.model_pipeline_dict[source]['func']
            kwargs = self.model_pipeline_dict[source]['args']
            self.pipe = f(**kwargs)
        
    def collect_response(self, message):

        if self.source == 'huggingface':
            raw_response = self.pipe(message, max_new_tokens=2048, temperature=self.temperature)
            message = raw_response[0]['generated_text']
            response = raw_response[0]['generated_text'][-1]['content']
        elif self.source in ['openai', 'deepseek', 'vec-inf']:
            raw_response = self.pipe.chat.completions.create(
                model=self.model,
                messages=message,
                temperature = self.temperature
            )
            response = raw_response.choices[0].message.content
            message.append({"role":'assistant', "content":response})
        elif self.source == 'google':
            curr_prompt = message[-1]['content']
            response = self.pipe.send_message(curr_prompt).text
            message.append({"role":"assistant", "content":response})
        
        return response, message
        
