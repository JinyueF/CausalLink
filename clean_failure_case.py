import json
import re
import os

def process_chat_history(jsonl_file, output_file):
    with open(jsonl_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Process each record (each line is a JSON object)
        for line in infile:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip invalid JSON lines
            
            history = data.get("conversational_history", [])
            cleaned_lines = []
            final_answer = None

            cause = data.get("cause")
            effect = data.get("effect")
            question = None
            if cause and effect:
                question = "Does {} moving cause {} to move?".format(cause, effect)
            
            history = data.get("conversational_history", [])
            cleaned_lines = []
            
            # Prepend the question if available.
            if question:
                cleaned_lines.append("Question: " + question)
            
            # 2. Process each message for actions and state updates.
            json_pattern = r'\{[\s\S]*?\}'
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "assistant":
                    # Use regex to extract the JSON portion.
                    match = re.search(json_pattern, content)
                    if match:
                        try:
                            response_data = json.loads(match.group())
                        except json.JSONDecodeError:
                            continue
                        # Process only if it contains an "action" and "shape".
                        if "action" in response_data and "shape" in response_data:
                            action = response_data["action"]
                            shape = response_data["shape"]
                            cleaned_lines.append("[{}] {}".format(action, shape))
                
                elif role == "user":
                    # Option 1: Look for states between
                    # "the current states of shapes are: " and "Based on the results you observe so far"
                    if "the current states of shapes are:" in content:
                        start_index = content.find("the current states of shapes are: ") + len("the current states of shapes are: ")
                        end_index = content.find("Based on the results you observe so far")
                        if end_index == -1:
                            end_index = len(content)
                        state_text = content[start_index:end_index].strip()
                        state_text = state_text.replace("\n", ", ")
                        cleaned_lines.append("State update: " + state_text)
                    # Option 2: Look for states between
                    # "Following are your current observations: " and "\n            Please interact with the shapes"
                    elif "Following are your current observations:" in content:
                        start_index = content.find("Following are your current observations: ") + len("Following are your current observations: ")
                        end_index = content.find("\n            Please interact with the shapes")
                        if end_index == -1:
                            end_index = len(content)
                        state_text = content[start_index:end_index].strip()
                        state_text = state_text.replace("\n", ", ")
                        cleaned_lines.append("State update: " + state_text)
            
            # 3. Extract the final answer from the assistant's message.
            #    Look for the assistant message containing a JSON with the "answer" key.
            for msg in reversed(history):
                if msg.get("role", "") == "assistant":
                    match = re.search(json_pattern, msg.get("content", ""))
                    if match:
                        try:
                            ans_data = json.loads(match.group())
                        except json.JSONDecodeError:
                            continue
                        if "answer" in ans_data:
                            final_answer = ans_data["answer"]
                            break
            
            # Prepend the question (if found) and append the final answer.
            if final_answer:
                cleaned_lines.append("Final Answer: " + final_answer)
            
            # Write out this record if any cleaned lines were produced.
            if cleaned_lines:
                outfile.write("\n".join(cleaned_lines) + "\n\n")

# Example usage:
if __name__ == '__main__':
    input_folder = "failure_cases"
    output_folder = "cleaned_failure_cases"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename.replace(".jsonl", ".txt"))
            process_chat_history(input_filepath, output_filepath)
            print(f"Processed {filename} -> {output_filepath}")
