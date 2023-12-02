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
RSSCN7_classes = ["aGrass",  "bField",  "cIndustry",  "dRiverLake",  "eForest",  "fResident",  "gParking"]
RSSCN7_label_text = {
                    "aGrass" : "This is the sound of creatures in a grassy land.",
                    "bField" : "This is the sound of an open field.",
                    "cIndustry": "This is the sound of industry or a manufacturing factory.",
                    "dRiverLake" : "This is the sound of a flowing river or the sound besides a lake or a water body.",
                    "eForest": "This is the sound that you can hear in forest where birds are chirping and wild animals live.",
                    "fResident": "This is the sound of a residential area where people are likely to be living.",
                    "gParking":"This is the sound of a parking space."
                    }

UCMerced_LandUse_classes = ['buildings', 'tenniscourt', 'harbor', 'airplane', 
                            'denseresidential', 'intersection', 'river',
                              'chaparral', 'beach', 'forest', 'agricultural',
                                'mobilehomepark', 'baseballdiamond', 'parkinglot',
                                  'golfcourse', 'storagetanks', 'mediumresidential',
                                    'freeway', 'sparseresidential', 'runway', 'overpass']
UCMerced_LandUse_label_text = {
                              "buildings":"This is the sound coming from buildings.",
                               "tenniscourt":"This is the sound heard from a tennis court where people are playing sports.",
                               "harbor":"This is the sound of a busy harbor where ships are docking.",
                                "airplane":"This is the sound of an airplane.",
                                "denseresidential":"This is the sound of a dense and crowded residential area.",
                                "intersection":"This is the sound likely to be heard in an intersection of an highway road.",
                                "river":"This is the sound of a flowing river with water.",
                                "chaparral":"This is the sound likely to be heard around chaparral where we can see vegetation consisting chiefly of tangled shrubs and thorny bushes in such areas.",
                                "beach":"This is the sound of waves in a sea beach.",
                                "forest":"This is the sound that you can hear in forest where birds are chirping and wild animals live.",
                                "agricultural":"This is the sound which is likely to be heard in an agricultural farm land.",
                                "mobilehomepark":"This is the sound of mobile home park where people are parking their home.",
                                "baseballdiamond":"This is the sound of a baselball sport where people are cheering.",
                                "parkinglot":"This is the sound of a parking lot where vehicles are being parked.",
                                "golfcourse":"This is the sound of a golf course.",
                                "storagetanks":"This is the sound of storage tanks.",
                                "mediumresidential":"This is the sound of a residential area with medium population.",
                                "freeway":"This is the sound of a freeway which is like highway but have higher speed limits.",
                                "sparseresidential":"This is the sound of a residential area with very less population. Porbably a suburban countryside.",
                                "runway":"This is the sound of a runway which is an airstrip, a (usually) paved section on which planes land or take off.",
                                "overpass":"This is the sound of an overpass which a crossing of two highways or of a highway and pedestrian path or railroad at different levels where clearance to traffic on the lower level is obtained by elevating the higher level."
                               }

def get_class_prompts(dataset_type):
    if dataset_type == "NWPU_RESISC45":
        text_dict = NWPU_RESISC45_label_text
    elif dataset_type == "RSSCN7":
        text_dict = RSSCN7_label_text
    elif dataset_type == "UCMerced_LandUse":
        text_dict = UCMerced_LandUse_label_text
    return text_dict

if __name__ == '__main__':
    print(set(NWPU_RESISC45_classes) == set(NWPU_RESISC45_label_text.keys()))
    print(set(RSSCN7_classes) == set(RSSCN7_label_text.keys()))
    print(set(UCMerced_LandUse_classes) == set(UCMerced_LandUse_label_text.keys()))
