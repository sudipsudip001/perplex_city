import json

file_path = "questions.json"

with open(file_path) as f:
    data = json.load(f)

new_data = []
for item in data:
    # Pop the question to separate it
    question = item.pop("question")
    # Wrap everything else in metadata
    new_data.append({"question": question, "metadata": item})

with open("questions_restructured.json", "w") as f:
    json.dump(new_data, f, indent=2)
