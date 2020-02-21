import json

person = '{"name": "Bob", "languages": ["English", "Fench"]}'
person_dict = json.loads(person)

# Output: {'name': 'Bob', 'languages': ['English', 'French']}
print( person_dict)

print("Hi, " + person_dict["name"])
print("You know " + person_dict['languages'][0] + " and " + person_dict['languages'][1] + " languages")
