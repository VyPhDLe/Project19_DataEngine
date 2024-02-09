import csv
import json


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
    # create a dictionary
    data = {}

    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to data
        for row in csvReader:
            # Assuming 'Loop' column to
            # be the primary key
            key = row['']
            if key not in data:
                data[key] = []
            data[key].append(row)

    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))



# Driver Code

# Decide the two file paths according to your
# computer system
csvFilePath = r'/Users/duyvy100701/PycharmProjects/Project19_DataEngine/Lab19_Dataset.csv'
jsonFilePath = r'/Users/duyvy100701/PycharmProjects/Project19_DataEngine/Lab19_Dataset.json'

# Call the make_json function
make_json(csvFilePath, jsonFilePath)
