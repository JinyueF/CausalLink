TEMPLATES = {
    'basic':{
        'system': 
            """You are in a world of shapes. The changes of shapes follow internal causal rules. 
            You are required to interact with the shapes and answer a causal question. 
            All changes in the world are deterministic and consistent. There is no hidden confounder. 
            
            Please provide a new JSON object for each of your reponse.""",
        'question':
            """Question: {}""",
        'initial':
            """Please propose your first interaction. Please provide your response by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes: {}
            - The value to "action" field must be one of the listed actions: {}

            {{
            "shape":"",
            "action":""
            }}
            """, 
        'choice':
            """Based on the results you observe so far, please decide to continue interaction or answer the question. 
            
            Please provide your response by filling JSON below:
            - The value to "next" field must be either "continue interaction" or "answer the question"

            {
            "next":""
            }
            """,
        'interaction':
            """Please propose your next interaction. Please provide your response by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes: {}
            - The value to "action" field must be one of the listed actions: {}

            {{
            "shape":"",
            "action":""
            }}
            """,
        'answer':
            """Please answer the question by filling the JSON below. 
        
            - The value to "answer" field must be "yes" or "no"

            {
            "answer":""
            }

            """
    }, 

    'basic_with_reason':{
        'system': 
            """You are in a world of shapes. The changes of shapes follow internal causal rules. 
            You are required to interact with the shapes and answer a causal question. 
            All changes in the world are deterministic and consistent. There is no hidden confounder. 
            
            Please provide a new JSON object for each of your reponse.""",
        'question':
            """Question: {}""",
        'initial':
            """Please reason step by step and propose your first interaction. Please provide your proposed interaction by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes: {}
            - The value to "action" field must be one of the listed actions: {}

            {{
            "shape":"",
            "action":""
            }}
            """, 
        'choice':
            """Based on the results you observe so far, please decide to continue interaction or answer the question. 
            
            Please provide your response by filling the JSON below:
            - The value to "next" field must be either "continue interaction" or "answer the question"

            {
            "next":""
            }
            """,
        'interaction':
            """Please reason step by step and propose your next interaction. Please provide your proposed interaction  by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes
            - The value to "action" field must be one of the listed actions

            {
            "shape":"",
            "action":""
            }
            """,
        'answer':
            """Please reason step by step and answer the question by filling the JSON below. 
        
            - The value to "answer" field must be "yes" or "no"

            {
            "answer":""
            }

            """
    }, 
    'basic_limit_steps': {
        'system': 
            """You are in a world of shapes. The changes of shapes follow internal causal rules. 
            You are required to interact with the shapes and answer a causal question. 
            All changes in the world are deterministic and consistent. There is no hidden confounder. 
            
            Please provide exactly one JSON object for each of your reponse. Try to reach an answer with the least amount of interactions.""",
        'question':
            """Question: {}""",
        'initial':
            """Please propose your first interaction. Please provide your response by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes: {}
            - The value to "action" field must be one of the listed actions: {}

            {{
            "shape":"",
            "action":""
            }}
            """, 
        'choice':
            """Based on the results you observe so far, please decide to continue interaction or answer the question. 
            
            Please provide your response by filling the JSON below:
            - The value to "next" field must be either "continue interaction" or "answer the question"

            {
            "next":""
            }
            """,
        'interaction':
            """Please propose your next interaction. Please provide your response by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes: {}
            - The value to "action" field must be one of the listed actions: {}

            {{
            "shape":"",
            "action":""
            }}
            """,
        'answer':
            """Please answer the question by filling the JSON below. 
        
            - The value to "answer" field must be "yes" or "no"

            {
            "answer":""
            }

            """
    }
}