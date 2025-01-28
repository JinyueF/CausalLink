TEMPLATES = {
    'basic':{
        'system': 
            """You are in a world of shapes. The movements of shapes follow internal causal rules. 
            You are required to interact with the shapes until you can answer a question about the causal rules. 
            All changes in the world are deterministic and consistent. There is no hidden confounder. 

            You can either 1) move a static shape or 2) hold a moving shape. A shape only stops moving when there is no other causes of its movement.
            """,
        
        'initial':
            """
            Following are your current observations: {}

            Please interact with the shapes to answer: {}
            
            Please propose your first interaction. Please provide your response by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes: {}
            - The value to "action" field must be one of the listed actions: {}

            {{
            "shape":"",
            "action":""
            }}
            """, 
        'choice':
            """Based on the results you observe so far, please decide to continue interaction or answer the question: {}. 
            
            Please provide your response by filling JSON below:
            - The value to "next" field must be either "continue interaction" or "answer the question"

            {{
            "next":""
            }}
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
            """
            You are ready to answer the question: {}

            Please answer the question by filling the JSON below. 
        
            - The value to "answer" field must be "yes" or "no"

            {{
            "answer":""
            }}

            """
    },
    'basic_limit_step':{
        'system': 
            """You are in a world of shapes. The movements of shapes follow internal causal rules. 
            You are required to interact with the shapes until you can answer a question about the causal rules. 
            All changes in the world are deterministic and consistent. There is no hidden confounder. 

            You can either 1) move a static shape or 2) hold a moving shape. A shape only stops moving when there is no other causes of its movement. Please reach the conclusion in the least number of steps possible. 
            """,
        
        'initial':
            """
            Following are your current observations: {}

            Please interact with the shapes to answer: {}
            
            Please propose your first interaction. Please provide your response by filling the JSON below:
            
            - The value to "shape" field must be one of the listed shapes: {}
            - The value to "action" field must be one of the listed actions: {}

            {{
            "shape":"",
            "action":""
            }}
            """, 
        'choice':
            """Based on the results you observe so far, please decide to continue interaction or answer the question: {}. 
            
            Please provide your response by filling JSON below:
            - The value to "next" field must be either "continue interaction" or "answer the question"

            {{
            "next":""
            }}
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
            """
            You are ready to answer the question: {}

            Please answer the question by filling the JSON below. 
        
            - The value to "answer" field must be "yes" or "no"

            {{
            "answer":""
            }}

            """
    }
}