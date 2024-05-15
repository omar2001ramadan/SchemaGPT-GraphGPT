In order to use "compare_jsons_to_excel, 
the prerequisite is to 

--------pip install openpyxl-------
--------pip install deepdiff-----

In the results comparison the closed brackets "{}" means there is no change in comparison
the open brackets "{" indicate there is a changed comaprison

----------TEST SET PROMPTS----------
These are all of the prompts used and their meanings:

---------------TEST SET 1------TESTING: GPT analysis is pretrained on stix models verbatim and must be rephrased ----------
STIX Verbatim Prompt:
“This scenario represents an advanced persistent threat (APT) intrusion set that is suspected to be funded by the country “Franistan”. Their target is the Branistan People’s Party (BPP), one of the political parties of the country “Branistan”. This intrusion set consists of a couple of sophisticated campaigns and attack patterns against the BPP’s website. One campaign seeks to insert false information into the BPP’s web pages, while another is a DDoS effort against the BPP web servers.”

STIX GPT4 Reshuffled Prompt:
“This case features a sophisticated intrusion set classified as an advanced persistent threat (APT), which is reportedly backed by Franistan, focusing on the Branistan People’s Party (BPP). This set comprises multiple complex attack strategies and campaigns directed at BPP’s website. Among these are efforts to infuse false data into BPP’s web pages and a separate DDoS attack targeting the party's web servers.”






--------------TEST SET 2--------TESTING: GPT analysis is pretrained on the STIX Scenarios and nouns used -----------
STIX Verbatim Noun Scrambling Prompt:
“This scenario represents an advanced persistent threat (APT) intrusion set that is suspected to be funded by the country “Maristan.”. Their target is the Liberty Party of Volandia (LPV), one of the political parties of the country “Volandia”. This intrusion set consists of a couple of sophisticated campaigns and attack patterns against the LPV’s website. One campaign seeks to insert false information into the LPV’s web pages, while another is a DDoS effort against the LPV web servers.”


STIX GPT4 Reshuffled & Noun Scrambling Prompt:
“This narrative describes a complex, advanced persistent threat (APT) breach operation likely supported by the nation "Maristan." The focus is on the Liberty Party of Volandia (LPV), a prominent political party in "Volandia." The breach operation involves various intricate maneuvers and initiatives aimed at the LPV’s website. One initiative involves disseminating misleading content across LPV’s web pages, while another consists of an overload attack designed to disrupt the LPV’s web servers.”








-----------------------TEST SET 3-------TESTING: GPT analysis is benefitted by GPT Phrasing, therefore, LLAMA3 has a different phrasing signature compared to GPT4 ------------------------
STIX Verbatim Prompt: "Same as above"

STIX LLAMA3 Reshuffled Prompt:
“The Branistan People's Party (BPP) finds itself in the crosshairs of a stealthy and highly sophisticated advanced persistent threat (APT) intrusion set, allegedly bankrolled by the mysterious forces of Franistan, which has unleashed a multi-pronged assault on the BPP's online stronghold, involving a clandestine campaign to insidiously inject disinformation into the party's website and a relentless distributed denial-of-service (DDoS) barrage aimed at crippling the BPP's web servers.”