# import lines
import requests
from bs4 import BeautifulSoup


# Grab url
r = requests.get("https://www.rottentomatoes.com/m/the_imitation_game/reviews")


soup = BeautifulSoup(r.text, "html.parser")

results = soup.findAll(class_="the_review")

print(results)
