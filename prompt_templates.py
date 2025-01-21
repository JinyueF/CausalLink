TEMPLATES = {
    'basic':{
        'system': 
            """You are in a world of shapes. The movements of shapes follow internal causal rules. 
            You are required to interact with the shapes and answer a causal question. 
            All changes in the world are deterministic and consistent. There is no hidden confounder. 

            You can move a static shape or hold a moving shape. A shape only stops moving when there
            is no other causes of its movement.
            """,
        'question':
            """Question: {}""",
        'initial':
            """
            Following are your current observations: {}
            
            Please propose your first interaction. Please provide your response by filling the JSON below:
            
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
    }
}