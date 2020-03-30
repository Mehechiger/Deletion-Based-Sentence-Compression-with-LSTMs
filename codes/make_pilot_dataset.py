import json

sent = "Serge Ibaka -- the Oklahoma City Thunder forward who was born in the Congo but played in Spain -- has been granted Spanish citizenship and will play for the country in EuroBasket this summer, the event where spots in the 2012 Olympics will be decided."

compression = "Serge Ibaka has been granted Spanish citizenship and will play in EuroBasket."

dataset = {1: {"sent": sent, "compression": compression}}

with open("../Google_dataset_news/pilot_dataset.json", "w") as f:
    json.dump(dataset, f)
