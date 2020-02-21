import json

# Here, we have used the open() function to read the json file.
# Then, the file is parsed using json.load() method which gives us a dictionary named data.

with open('person_data.json') as f:
  data = json.load(f)

print(data)

print(data['name'])
print(data['languages'])
