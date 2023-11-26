import json

# Load the JSON data from the file
with open(r'TechQA\training_and_dev\training_Q_A.json','r') as file:
    data = json.load(file)

# Extract QUESTION_TEXT and ANSWER for each entry and format as desired
formatted_data = []
for entry in data:
    question_text = entry.get("QUESTION_TEXT", "")
    answer = entry.get("ANSWER", "")

    # Create a dictionary with "input" and "output" keys
    formatted_entry = {"input": question_text, "output": answer}
    
    # Append the formatted entry to the list
    formatted_data.append(formatted_entry)

# Write the formatted data to a JSON Lines (jsonl) file
with open('formatted_data.jsonl', 'w') as outfile:
    for entry in formatted_data:
        # Convert each dictionary to a JSON-formatted string and write to the file
        json.dump(entry, outfile)
        outfile.write('\n')  # Add a newline character after each entry
