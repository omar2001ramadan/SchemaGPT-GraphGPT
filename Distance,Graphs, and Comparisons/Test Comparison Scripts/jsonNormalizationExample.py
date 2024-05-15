import spacy
import json
import re

# Load the spaCy model for English
nlp = spacy.load('en_core_web_sm')

# Sample JSON data (you can load this from a file or directly paste it into the code)
json_data = [
    # The first JSON data block (Threat actors)
    [
        {
            "type": "threat-actor",
            "id": "threat-actor--1",
            "name": "Disco Team Threat Actor Group",
            "description": "This organized threat actor group operates to create profit from all types of crime.",
            "aliases": ["Equipo del Discoteca"],
            "goals": ["Steal Credit Card Information"],
            "sophistication": "expert",
            "resource_level": "organization",
            "primary_motivation": "personal-gain"
        },
        {
            "type": "identity",
            "id": "identity-2",
            "name": "Disco Team",
            "description": "Disco Team is the name of an organized threat actor crime-syndicate.",
            "identity_class": "organization",
            "contact_information": "disco-team@stealthemail.com"
        },
        {
            "type": "relationship",
            "id": "relationship--3",
            "relationship_type": "attributed-to",
            "source_ref": "threat-actor--1",
            "target_ref": "identity--2"
        }
    ]
    # The second JSON data block (APT Intrusion Set)
]

# Function to normalize text: lowercase, remove special characters, and lemmatize
def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters except spaces and alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Lemmatize the cleaned text
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])
    return lemmatized_text

# Function to process each JSON block
def process_json(json_block):
    for entry in json_block:
        for key, value in entry.items():
            if isinstance(value, str):  # Normalize all string fields
                entry[key] = normalize_text(value)
            elif isinstance(value, list):  # Normalize lists of strings (like aliases)
                entry[key] = [normalize_text(item) if isinstance(item, str) else item for item in value]
    return json_block

# Process each block of JSON data
processed_data = [process_json(block) for block in json_data]

# Print or save the processed data
print(json.dumps(processed_data, indent=2))