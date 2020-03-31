import json

sents = ["Serge Ibaka -- the Oklahoma City Thunder forward who was born in the Congo but played in Spain -- has been granted Spanish citizenship and will play for the country in EuroBasket this summer, the event where spots in the 2012 Olympics will be decided.",
         "MILAN -Catania held Roma to a 1-1 draw in Serie A on Wednesday as the teams played out the remaining 25 minutes of a game that was called off last month.",
         "tate Street Corporation, a provider of investment servicing, investment management and investment research and trading services, has launched a new investment servicing solution to support small to mid-sized asset managers with their investment operations needs.",
         "Massey Energy, the fourth largest coal company in the country, could have its corporate charter revoked if public interest organizations have their way.",
         "Foundation for MetroWest has awarded $55,000 in grants to 37 food pantries across the region, drawing on two of its funds that provide winter assistance to families in need.",
         "Wong won't rule out super raid Updated: 08:17, Tuesday March 26, 2013 Finance Minister Penny Wong has declined to rule out axing tax concessions for superannuation in the May budget.",
         "TINY 'robots' that can be dropped into an industrial process to report any problems are being developed by a Woolston company.",
         "Jordan and South Korea have pledged to cooperate on nuclear regulatory issues as part of an agreement signed on Saturday.",
         "Just after the Fourth of July holiday, and only a week after the city's budget has been completed, City Council Member Robert Jackson will hold two town hall meetings in cooperation with several other Northern Manhattan elected officials.",
         "Tim Cowlishaw made a bad joke and now the MMA blogosphere is lighting torches and sharpening pitchforks."
         ]

comprs = ["Serge Ibaka has been granted Spanish citizenship and will play in EuroBasket.",
          "Catania held Roma to a 1 1 draw in Serie A.",
          "State Street Corporation, has launched a new investment servicing solution.",
          "Massey Energy, the coal company, could have its corporate charter revoked.",
          "Foundation for MetroWest has awarded $ 55,000 in grants to 37 food pantries across the region.",
          "Wong won't rule out super raid.",
          "TINY robots are being developed by a company.",
          "Jordan and South Korea have pledged to cooperate on regulatory issues.",
          "Robert Jackson will hold two town hall meetings.",
          "Tim Cowlishaw made a bad joke."
          ]

dataset = [[sents[i], comprs[i]] for i in range(len(sents))]

with open("../Google_dataset_news/pilot3_dataset.json", "w") as f:
    json.dump(dataset, f)
