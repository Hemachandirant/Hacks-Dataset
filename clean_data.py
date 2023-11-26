import json
import re

def clean_text(text):
    # Remove '\n' characters
    text = text.replace('\n', ' ')

    # Remove extra whitespaces
    text = ' '.join(text.split())

    # Add any other specific characters you want to remove

    return text

# Load the JSON data from the file
with open(r'TechQA\training_and_dev\training_Q_A.json', 'r') as file:
    data = json.load(file)

# Clean the text for each entry
cleaned_data = []
for entry in data:
    question_text = clean_text(entry.get("QUESTION_TEXT", ""))
    answer = clean_text(entry.get("ANSWER", ""))

    # Create a dictionary with "input" and "output" keys
    cleaned_entry = {"input": question_text, "output": answer}

    # Append the cleaned entry to the list
    cleaned_data.append(cleaned_entry)

# Write the cleaned data to a JSON Lines (jsonl) file
with open('cleaned_data.jsonl', 'w') as outfile:
    for entry in cleaned_data:
        # Convert each dictionary to a JSON-formatted string and write to the file
        json.dump(entry, outfile)
        outfile.write('\n')  # Add a newline character after each entry
