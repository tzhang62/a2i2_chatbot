
import re
import json

input_file = '/Users/tzhang/ParlAI/parlai/scripts/train_data_output.txt'

def convert_to_target_format(dialogues):
    results = []

    for dialogue in dialogues.strip().split("- - -EPISODE- - -"):
        if not dialogue.strip():
            continue

        lines = dialogue.strip().split("\n")

        # Extract persona information
        persona_lines = [line for line in lines if line.startswith("your persona:")]
        persona = " ".join(line.replace("your persona: ", "").strip() for line in persona_lines)

        # Extract the actual conversation lines (non-persona lines)
        conversation_lines = [line for line in lines if not line.startswith("your persona:")]
        answers = conversation_lines[-1].strip()
        conversation_lines = conversation_lines[:-1]

        # Create context from the first utterance and answer from the last utterance
        context_lines = ''
        if conversation_lines:
            for i in range(len(conversation_lines)):
                if i % 2 == 0:
                    utter = "Q: " + conversation_lines[i].strip()
                    context_lines+=utter
                else:
                    utter = "  R:" + conversation_lines[i].strip()
                    context_lines+=utter
                    



            question = re.sub(r'^.*?\d+ - - -\n', '', context_lines).strip()
        else:
            question = ""
            answers = ""

        # Construct the result dictionary
        result = {
            "question": f"persona: {persona} context: {question}",
            "answers": [answers]
        }
        results.append(result)

    return results


# Read dialogues from a text file
def read_dialogues_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

dialogues = read_dialogues_from_file(input_file)


# Convert the dialogues and save as JSONL file
converted_data = convert_to_target_format(dialogues)
converted_data_100 = converted_data[:10000]

# Save to a JSONL file
output_file = "/Users/tzhang/Documents/LAPDOG_new_dataset/train_data_10000.jsonl"
with open(output_file, "w") as f:
    for entry in converted_data_100:
        f.write(json.dumps(entry) + "\n")
import pdb; pdb.set_trace()
