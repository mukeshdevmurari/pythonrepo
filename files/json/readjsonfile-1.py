import json

with open('jsondata.json') as f:
    data = json.load(f)

print(data)

students = data['students']

for student in students:
    print(student["name"] + " - " + student["city"])
