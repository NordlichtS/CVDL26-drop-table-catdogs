"""
Description for the api key acknowledment:
App which determines if on the picture is a cat and defines the breed of the cat. 
We use the dataset to train our machine learning and computer vision model for a project at university, 
which we then want to extend to further animals and later on extend this learning algorithm to identify if this specific animal 
was already seen once and categorize it. 
This will help wildforst and nationalparks to learn about their animals and paths they have taken at scale with the help of tourist pictures, 
which they can upload at a local datacenter / cloud. 
The rangers work should be reduced and automated so they can overview the paths and animals at scale 
"""

import yaml
import requests

if __name__ =="__main__":
    stream = open("api_keys.yaml", "r")
    dictionary = yaml.load(stream)
    for key, value in dictionary.items():
        print (key + ": " + str(value))



limit= 100
params = [{
    "id":"ebv",
    "has_breeds": 1,
    "page": 0
}]
headers = {"x-api-key": stream.cat_api.x_api_key}
url = stream.cat_api.url


response= requests.get(url, headers= headers, params=params)
if response.status_code == 200:
    cat_data = response.json()
    for cat in cat_data:
        print(cat["url"])
else:
    print(f"Failed to grab data: {response.status_code}")

