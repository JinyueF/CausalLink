TEMPLATES = {
    'basic':{
        'system': 
            """You are in a world of shapes. The changes of shapes follow internal causal rules. 
            You are required to interact with the shapes and answer a causal question. 
            All changes in the world are deterministic and consistent. There is no hidden confounder. """,
        'question':
            """Question: {}
            Shapes: {}
            Actions: {}""",
        'initial':
            """Please reason about the question and propose your first interaction. Please provide your response in JSON format:
            
            - The value to "shape" field must be one of the listed shapes
            - The value to "action" field must be one of the listed actions

            {
            "shape":"",
            "action":""
            }
            """, 
        'choice':
            """Based on the results you observe so far, please decide to continue interaction or answer the question. 
            
            Please provide your response in JSON format:
            - The value to "next" field must be either "continue interaction" or "answer the question"

            {
            "next":""
            }
            """,
        'interaction':
            """Please reason about the question and propose your next interaction. Please provide your response in JSON format:
            
            - The value to "shape" field must be one of the listed shapes
            - The value to "action" field must be one of the listed actions

            {
            "shape":"",
            "action":""
            }
            """,
        'answer':
            """Please answer the question {} in JSON format. 
        
            - The value to "answer" field must be "yes" or "no"

            {
            "answer":""
            }

            """
    }
}