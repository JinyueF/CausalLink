TEMPLATES = {
    'no_reasoning': {
        'system': 
            "You are in a world of shapes. The changes of shapes follow internal causal rules. \
            You are required to interact with the shapes and answer a causal question. \
            All changes in the world are deterministic and consistent. There is no hidden confounder. ",
            'question':
            "Question: {} \n \
            Shape: {} \n \
            Action: {} \n",
        'json_format':
            """Please provide your response for your chosen interaction in JSON format:

            - The value to "shape" field must be one of the listed shapes
            - The value to "action" field must be one of the listed actions
            - If you are ready, answer the question with "yes" or "no" in the field "answer". \
                Otherwise, fill the field with "next interaction"

                {
                "shape": "",
                "action": "",
                "answer": ""
                }

            """}, 

    'requires_reasoning':{
        'system': 
            "You are in a world of shapes. The changes of shapes follow internal causal rules. \
            You are required to interact with the shapes and answer a causal question. \
            All changes in the world are deterministic and consistent. There is no hidden confounder. ",
            'question':
            "Question: {} \n \
            Shape: {} \n \
            Action: {} \n",
        'json_format':
            """Please reason about the interactions and provide your response for your chosen interaction in JSON format:

            - The value to "shape" field must be one of the listed shapes
            - The value to "action" field must be one of the listed actions
            - If you are ready, answer the question with "yes" or "no" in the field "answer". \
                Otherwise, fill the field with "next interaction"
            - Explain your reasoning steps in the "reasoning" field

                {
                "shape": "",
                "action": "",
                "answer": "",
                "reasoning":""
                }

            """
    }
}