dependencies->
pip install spacy
python -m spacy download en_core_web_sm



Comparison strucuture
Step 1: Key/value Normalization
	lowecase text conversion
	eliminate unecessary charachters  such as hyphen (Franistan-backed APT) with "franistan backed apt"

step 2: reducing the semantics of words:
	
	lemmatization (reducing word to root form)
	named entity recodnition (NER) #unnecessary for fake countries and names but useful for real datasets
	POS tagging (not directly useful but can possibly help NER by giving context to phrase verbage and syntax)

step 3:


EXAMPLE: 
schema_examples.json
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

NORMALIZED:
schema_examples.json
[
  [
    {
      "type": "threatactor",
      "id": "threatactor1",
      "name": "disco team threat actor group",
      "description": "this organize threat actor group operate to create profit from all type of crime",
      "aliases": [
        "equipo del discoteca"
      ],
      "goals": [
        "steal credit card information"
      ],
      "sophistication": "expert",
      "resource_level": "organization",
      "primary_motivation": "personalgain"
    },
    {
      "type": "identity",
      "id": "identity2",
      "name": "disco team",
      "description": "disco team be the name of an organize threat actor crimesyndicate",
      "identity_class": "organization",
      "contact_information": "discoteamstealthemailcom"
    },
    {
      "type": "relationship",
      "id": "relationship3",
      "relationship_type": "attributedto",
      "source_ref": "threatactor1",
      "target_ref": "identity2"
    }
  ]
]

