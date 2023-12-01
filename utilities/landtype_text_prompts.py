#This script is responsible for returning the list of textual prompts ideal for zero-shot classification of land-cover types.

NWPU_RESISC45_classes = ["Airfield", "Anchorage", "Beach", "Dense-Residential", "Farm",  "Flyover",  "Forest",  "Game-Space",  "Parking-Space" , "River" , "Sparse-Residential",  "Storage-Cisterns"]
NWPU_RESISC45_label_text = {
    "Airfield": "This is the sound of an airfield.",
    "Anchorage": "This is the sound of an anchorage which also could mean dock, harbor, haven, port, refuge, and roadstead.",
    "Beach": "This is the sound of waves in a sea beach.",
    "Dense-Residential" : "This is the sound of dense residential area.",
    "Farm": "This is the sound of a farm.",
    "Flyover": "This is the sound of a flyover.",
    "Forest": "This is the sound of a forest.",
    "Game-Space": "This is the sound of a game space where sporting event is happening.",
    "Parking-Space": "This is the sound of a parking space where cars, bus could have been parked.",
    "River":"This is the sound of flowing river.",
    "Sparse-Residential": "This is the sound of sparse residential area with less population.",
    "Storage-Cisterns": "This is the sound of Water storage tanks, also known as cisterns, are primarily used to store water for domestic and consumptive purposes in households or buildings."
                            }

def get_class_prompts(dataset_type):

    if dataset_type == "NWPU_RESISC45":
        text_dict = NWPU_RESISC45_label_text
    elif dataset_type == "RSSCN7":
        RSSCN7_classes = ["aGrass",  "bField",  "cIndustry",  "dRiverLake",  "eForest",  "fResident",  "gParking"]#images in "*.jpg" format.

        RSSCN7_label_text = {
            "aGrass" : "This is the sound of creatures in a grassy land."
            "bField" : "This is the sound of creatures in an open field."
        }

    return text_dict
