"""
Q&A 生成器 4: food_*, technology_*, calendar_time, weather, colors_visual
"""

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "qa"
OUT_DIR.mkdir(exist_ok=True)

random.seed(45)

def write_jsonl(category: str, items: list):
  path = OUT_DIR / f"{category}.jsonl"
  with path.open('w', encoding='utf-8') as f:
  for it in items:
  f.write(json.dumps(it, ensure_ascii=False) + '\n')
  print(f"[{category}] {len(items)} pairs -> {path}")

# ---------------------------------------------------------------------------
# food_basics
# ---------------------------------------------------------------------------

FOOD_FACTS = [
  ("What is pizza made of?", "Dough, sauce, and cheese."),
  ("What is bread made of?", "Flour, water, yeast, salt."),
  ("What is pasta made of?", "Flour, water, sometimes eggs."),
  ("What is rice?", "A grain we cook and eat."),
  ("What is sushi?", "Rice with fish or veggies."),
  ("What is tempura?", "Battered and fried food."),
  ("What is ramen?", "Japanese noodle soup."),
  ("What is udon?", "Thick Japanese noodles."),
  ("What is soba?", "Japanese buckwheat noodles."),
  ("What is curry?", "Spiced stew with meat or veg."),
  ("What is sashimi?", "Sliced raw fish."),
  ("What is miso?", "Fermented soybean paste."),
  ("What is tofu?", "Bean curd made from soy."),
  ("What is cheese?", "A food made from milk."),
  ("What is butter?", "A solid fat from milk."),
  ("What is yogurt?", "Fermented milk product."),
  ("What is honey?", "Sweet syrup made by bees."),
  ("What is sugar?", "Sweetener from cane or beet."),
  ("What is salt?", "Sodium chloride mineral."),
  ("What is pepper?", "Spice from peppercorns."),
  ("What is chocolate?", "Treat made from cocoa beans."),
  ("What is ice cream?", "Frozen sweet dairy treat."),
  ("What is cake?", "Sweet baked dessert."),
  ("What is a pie?", "Pastry with filling inside."),
  ("What is a sandwich?", "Food between bread slices."),
  ("What is a burger?", "Patty of meat in a bun."),
  ("What is a hot dog?", "Sausage in a long bun."),
  ("What is a taco?", "Corn or flour shell with fillings."),
  ("What is a burrito?", "Tortilla wrapped with fillings."),
  ("What is a salad?", "Mixed raw veggies, often dressed."),
  ("What is soup?", "A liquid food, often hot."),
  ("What is a steak?", "A thick cut of meat."),
  ("What is bacon?", "Cured strips of pork."),
  ("What is ham?", "Cured leg of pork."),
  ("What is a chicken wing?", "A small cut of chicken."),
  ("What is a meatball?", "Ball of ground meat."),
  ("What is jam?", "Sweet spread of cooked fruit."),
  ("What is jelly?", "Clear fruit gel spread."),
  ("What is peanut butter?", "Paste made from peanuts."),
  ("What is mustard?", "Yellow spicy condiment."),
  ("What is ketchup?", "Tomato sauce in a bottle."),
  ("What is mayo?", "Mayonnaise, egg-based sauce."),
  ("What is olive oil?", "Oil pressed from olives."),
  ("What is wine?", "Fermented grape juice."),
  ("What is beer?", "Fermented grain drink."),
  ("What is coffee?", "Drink from roasted beans."),
  ("What is tea?", "Drink from tea leaves."),
  ("What is milk?", "Liquid from cows or plants."),
  ("What is juice?", "Pressed liquid from fruit."),
  ("What is soda?", "A sweet fizzy drink."),
  ("Where does pasta come from?", "From Italy."),
  ("Where does sushi come from?", "From Japan."),
  ("Where does tea originally come from?", "From China."),
  ("Where does coffee come from?", "From Ethiopia originally."),
  ("Where does chocolate come from?", "From cacao trees."),
  ("Where do potatoes come from?", "From South America."),
  ("Where do tomatoes come from?", "From the Americas."),
  ("Where does cheese come from?", "Many countries make it."),
  ("Is a tomato a fruit?", "Yes, it is a fruit."),
  ("Is an apple a fruit?", "Yes, apple is a fruit."),
  ("Is broccoli a vegetable?", "Yes, it is a vegetable."),
  ("What is a banana?", "A long yellow fruit."),
  ("What color is a banana?", "Yellow when ripe."),
  ("What is a strawberry?", "A small red fruit."),
  ("What is a watermelon?", "A big green fruit, juicy inside."),
  ("What is a lemon?", "Sour yellow citrus fruit."),
  ("What is a lime?", "Sour green citrus fruit."),
  ("What is an orange?", "Sweet orange citrus fruit."),
  ("What is a grape?", "Small round sweet fruit."),
  ("What is a peach?", "Soft sweet fruit with fuzzy skin."),
  ("What is a mango?", "Sweet tropical orange fruit."),
  ("What is a pineapple?", "Tropical fruit with spiky skin."),
  ("What is a pear?", "Sweet fruit with green skin."),
  ("What is a cherry?", "Small red round fruit."),
  ("What is a melon?", "A large round juicy fruit."),
  ("Why eat vegetables?", "They have vitamins and fiber."),
]

def gen_food_basics():
  items = 
  for q, a in FOOD_FACTS:
  items.append({"category": "food_basics", "q": q, "a": a})
  for q, a in FOOD_FACTS:
  items.append({"category": "food_basics",
  "q": "Food: " + q.lower(),
  "a": a})
  write_jsonl("food_basics", items[:150])

# ---------------------------------------------------------------------------
# food_origins
# ---------------------------------------------------------------------------

FOOD_ORIGINS = [
  ("Where does sushi come from?", "From Japan."),
  ("Where does pizza come from?", "From Italy."),
  ("Where does pasta come from?", "From Italy."),
  ("Where does paella come from?", "From Spain."),
  ("Where does croissant come from?", "From France."),
  ("Where does kebab come from?", "From the Middle East."),
  ("Where does hummus come from?", "From the Middle East."),
  ("Where does falafel come from?", "From the Middle East."),
  ("Where does curry come from?", "Largely from India."),
  ("Where does naan come from?", "From India."),
  ("Where does samosa come from?", "From India."),
  ("Where does dim sum come from?", "From China."),
  ("Where does dumpling come from?", "From China."),
  ("Where does ramen come from?", "From Japan, China origins."),
  ("Where does pho come from?", "From Vietnam."),
  ("Where does pad thai come from?", "From Thailand."),
  ("Where does kimchi come from?", "From Korea."),
  ("Where does bulgogi come from?", "From Korea."),
  ("Where does taco come from?", "From Mexico."),
  ("Where does burrito come from?", "From Mexico."),
  ("Where does empanada come from?", "From Spain and Latin America."),
  ("Where does feijoada come from?", "From Brazil."),
  ("Where does churro come from?", "From Spain."),
  ("Where does waffle come from?", "From Belgium."),
  ("Where does poutine come from?", "From Canada."),
  ("Where does pierogi come from?", "From Poland."),
  ("Where does goulash come from?", "From Hungary."),
  ("Where does fondue come from?", "From Switzerland."),
  ("Where does schnitzel come from?", "From Austria."),
  ("Where does bratwurst come from?", "From Germany."),
  ("Where does fish and chips come from?", "From the UK."),
  ("Where does shepherd's pie come from?", "From the UK."),
  ("Where does haggis come from?", "From Scotland."),
  ("Where does moussaka come from?", "From Greece."),
  ("Where does gyro come from?", "From Greece."),
  ("Where does tiramisu come from?", "From Italy."),
  ("Where does gelato come from?", "From Italy."),
  ("Where does macarons come from?", "From France."),
  ("Where does baklava come from?", "From Turkey."),
  ("Where does sauerkraut come from?", "From Germany."),
]

def gen_food_origins():
  items = 
  for q, a in FOOD_ORIGINS:
  items.append({"category": "food_origins", "q": q, "a": a})
  for q, a in FOOD_ORIGINS:
  items.append({"category": "food_origins",
  "q": "Origin of " + q.split("Where does ")[1].replace(" come from?", "?"),
  "a": a})
  write_jsonl("food_origins", items[:80])

# ---------------------------------------------------------------------------
# technology_basics
# ---------------------------------------------------------------------------

TECH_BASICS = [
  ("Who founded Microsoft?", "Bill Gates and Paul Allen."),
  ("Who founded Apple?", "Steve Jobs, Wozniak, Wayne."),
  ("Who founded Google?", "Larry Page and Sergey Brin."),
  ("Who founded Amazon?", "Jeff Bezos founded it."),
  ("Who founded Facebook?", "Mark Zuckerberg and friends."),
  ("Who founded Tesla?", "Martin Eberhard and Tarpenning."),
  ("Who founded SpaceX?", "Elon Musk founded it."),
  ("Who founded Twitter?", "Jack Dorsey and others."),
  ("Who founded YouTube?", "Steve Chen, Chad Hurley, Karim."),
  ("Who founded Netflix?", "Reed Hastings and Marc Randolph."),
  ("Who founded Uber?", "Travis Kalanick and Garrett Camp."),
  ("Who founded Airbnb?", "Brian Chesky and others."),
  ("Who founded LinkedIn?", "Reid Hoffman and others."),
  ("Who founded Instagram?", "Kevin Systrom and Krieger."),
  ("Who founded WhatsApp?", "Jan Koum and Brian Acton."),
  ("Who founded Spotify?", "Daniel Ek and Lorentzon."),
  ("Who founded TikTok?", "Zhang Yiming founded ByteDance."),
  ("Who founded Snapchat?", "Evan Spiegel and partners."),
  ("Who founded Reddit?", "Steve Huffman and Aaron Swartz."),
  ("Who founded Wikipedia?", "Jimmy Wales and Larry Sanger."),
  ("What is HTML?", "Markup for web pages."),
  ("What is CSS?", "Style language for web pages."),
  ("What is JavaScript?", "Language to run web logic."),
  ("What is Python?", "A popular programming language."),
  ("What is Java?", "A widely used programming language."),
  ("What is C++?", "A fast systems language."),
  ("What is the internet?", "Global network of computers."),
  ("What is the World Wide Web?", "Web of pages on the internet."),
  ("What is an URL?", "A web address like example.com."),
  ("What is a browser?", "Software to view web pages."),
  ("Name a web browser.", "Chrome, Firefox, Safari."),
  ("What is Chrome?", "Google's web browser."),
  ("What is Firefox?", "Mozilla's web browser."),
  ("What is Safari?", "Apple's web browser."),
  ("What is an OS?", "Operating system like Windows."),
  ("Name an OS.", "Windows, macOS, Linux."),
  ("What is Windows?", "Microsoft's OS for PCs."),
  ("What is macOS?", "Apple's OS for Macs."),
  ("What is Linux?", "Open source OS family."),
  ("What is iOS?", "Apple's mobile OS."),
  ("What is Android?", "Google's mobile OS."),
  ("What is a CPU?", "Central processor of a computer."),
  ("What is RAM?", "Memory used while running."),
  ("What is a hard drive?", "Storage for files long-term."),
  ("What is SSD?", "Solid state drive, fast storage."),
  ("What is a GPU?", "Graphics processor unit."),
  ("What is WiFi?", "Wireless network technology."),
  ("What is Bluetooth?", "Short range wireless link."),
  ("What is 5G?", "Fifth generation mobile network."),
  ("What is AI?", "Artificial intelligence."),
  ("What is machine learning?", "Software that learns from data."),
  ("What is deep learning?", "ML with deep neural networks."),
  ("What is a neural network?", "Layers of weighted units."),
  ("What is ChatGPT?", "An AI chatbot by OpenAI."),
  ("What is Claude?", "An AI assistant by Anthropic."),
  ("What is open source?", "Software with public code."),
  ("What is GitHub?", "A site to share code."),
  ("What is Git?", "Version control software."),
  ("What is a file?", "A unit of stored data."),
  ("What is a folder?", "A container for files."),
  ("What is encryption?", "Hiding data with a key."),
  ("What is a password?", "A secret used to log in."),
  ("What is a hacker?", "Someone who studies systems."),
  ("What is malware?", "Bad software like viruses."),
  ("What is a virus?", "A program that spreads itself."),
  ("What is a server?", "A computer serving requests."),
  ("What is a client?", "A computer making requests."),
  ("What is cloud computing?", "Computing on remote servers."),
  ("What is a database?", "An organized store of data."),
  ("What is SQL?", "Language to query databases."),
  ("What is a smartphone?", "A pocket computer with phone."),
  ("Who invented the computer?", "Many people, including Babbage."),
  ("Who invented the World Wide Web?", "Tim Berners-Lee."),
  ("Who is Alan Turing?", "Father of modern computer science."),
  ("Who is Linus Torvalds?", "Creator of Linux."),
  ("Who is Bjarne Stroustrup?", "Creator of C++."),
  ("Who is Guido van Rossum?", "Creator of Python."),
  ("Who is Brendan Eich?", "Creator of JavaScript."),
  ("What is binary?", "Numbers made of 0 and 1."),
  ("What is a bit?", "A 0 or 1, smallest data unit."),
  ("What is a byte?", "Eight bits together."),
  ("How many bits in a byte?", "Eight bits."),
]

def gen_technology_basics():
  items = 
  for q, a in TECH_BASICS:
  items.append({"category": "technology_basics", "q": q, "a": a})
  for q, a in TECH_BASICS:
  items.append({"category": "technology_basics",
  "q": "Tech: " + q,
  "a": a})
  write_jsonl("technology_basics", items[:150])

# ---------------------------------------------------------------------------
# technology_programming
# ---------------------------------------------------------------------------

PROG_QA = [
  ("What is a variable?", "A named storage for a value."),
  ("What is a loop?", "Code that repeats a block."),
  ("What is a function?", "A reusable block of code."),
  ("What is a class?", "A template for objects."),
  ("What is an object?", "An instance of a class."),
  ("What is an array?", "An ordered list of values."),
  ("What is a string?", "A sequence of characters."),
  ("What is an integer?", "A whole number."),
  ("What is a float?", "A decimal number."),
  ("What is a boolean?", "True or false value."),
  ("What is a list in Python?", "An ordered mutable array."),
  ("What is a tuple?", "An immutable ordered list."),
  ("What is a dictionary?", "Key to value mapping."),
  ("What is a set?", "Collection of unique items."),
  ("What is recursion?", "A function calling itself."),
  ("What is a bug?", "An error in the code."),
  ("What is debugging?", "Finding and fixing bugs."),
  ("What is a compiler?", "Tool to translate code."),
  ("What is an interpreter?", "Tool to run code directly."),
  ("What is a syntax error?", "Wrong grammar in code."),
  ("What is a runtime error?", "An error while running."),
  ("What is IDE?", "Integrated dev environment."),
  ("What is VS Code?", "A popular code editor."),
  ("What is Python used for?", "Scripts, AI, web, more."),
  ("What is C used for?", "Systems and embedded code."),
  ("What is HTML used for?", "Structuring web pages."),
  ("What is CSS used for?", "Styling web pages."),
  ("What is JavaScript used for?", "Logic in web pages."),
  ("What is an API?", "Application programming interface."),
  ("What is JSON?", "Text format for data."),
  ("What is XML?", "Markup format like HTML."),
  ("What is a server-side language?", "Code run on the server."),
  ("What is a client-side language?", "Code run in the browser."),
  ("What is Node.js?", "JavaScript run outside browser."),
  ("What is React?", "JS library by Facebook."),
  ("What is Vue?", "JS framework, simple style."),
  ("What is Django?", "Python web framework."),
  ("What is Flask?", "Lightweight Python web tool."),
  ("What is FastAPI?", "Modern Python web tool."),
  ("What is a framework?", "A reusable code platform."),
  ("What is a library?", "Reusable code you import."),
  ("What is open source?", "Code anyone can read."),
  ("What is unit test?", "A test for small code parts."),
  ("What is integration test?", "Test of parts together."),
  ("What is version control?", "Tracking code changes."),
  ("What is Git?", "Common version control tool."),
  ("What is GitHub?", "Cloud Git host."),
  ("What is a branch?", "A line of code work."),
  ("What is merge?", "Combine two branches."),
  ("What is a commit?", "A saved change in Git."),
]

def gen_technology_programming():
  items = 
  for q, a in PROG_QA:
  items.append({"category": "technology_programming", "q": q, "a": a})
  for q, a in PROG_QA:
  items.append({"category": "technology_programming",
  "q": "Code: " + q.lower(),
  "a": a})
  write_jsonl("technology_programming", items[:100])

# ---------------------------------------------------------------------------
# calendar_time
# ---------------------------------------------------------------------------

TIME_QA = [
  ("How many days in a year?", "365 days, 366 in a leap year."),
  ("How many days in a leap year?", "366 days."),
  ("How many days in a month?", "28 to 31 days."),
  ("How many days in February?", "28 or 29 days."),
  ("How many days in April?", "30 days."),
  ("How many days in July?", "31 days."),
  ("How many months in a year?", "12 months."),
  ("How many weeks in a year?", "About 52 weeks."),
  ("How many days in a week?", "7 days."),
  ("How many hours in a day?", "24 hours."),
  ("How many minutes in an hour?", "60 minutes."),
  ("How many seconds in a minute?", "60 seconds."),
  ("How many seconds in an hour?", "3600 seconds."),
  ("How many minutes in a day?", "1440 minutes."),
  ("How many seconds in a day?", "86,400 seconds."),
  ("How many hours in a week?", "168 hours."),
  ("Name the months in order.", "January through December."),
  ("Name the days of the week.", "Sunday through Saturday."),
  ("What is a decade?", "Ten years."),
  ("What is a century?", "One hundred years."),
  ("What is a millennium?", "One thousand years."),
  ("What is a year?", "Earth's orbit around the sun."),
  ("What is a month?", "Roughly the moon's cycle."),
  ("What is a day?", "One Earth rotation."),
  ("What is noon?", "12 PM, middle of the day."),
  ("What is midnight?", "12 AM, middle of the night."),
  ("What is AM?", "Before noon."),
  ("What is PM?", "After noon."),
  ("How many days in March?", "31 days."),
  ("How many days in June?", "30 days."),
  ("How many days in September?", "30 days."),
  ("How many days in October?", "31 days."),
  ("How many days in November?", "30 days."),
  ("How many days in December?", "31 days."),
  ("How many days in January?", "31 days."),
  ("How many days in May?", "31 days."),
  ("How many days in August?", "31 days."),
  ("What month has 28 days?", "February sometimes 29."),
  ("What is the first day of the week?", "Often Sunday or Monday."),
  ("What is the last day of the week?", "Saturday in many places."),
  ("What is the weekend?", "Saturday and Sunday."),
  ("What is a weekday?", "Monday through Friday."),
  ("What is a leap year?", "A year with an extra day."),
  ("When is the next leap year after 2024?", "In 2028."),
  ("Why do leap years exist?", "To keep calendar aligned."),
  ("What is a quarter of a year?", "Three months."),
  ("How many quarters in a year?", "Four quarters."),
  ("How many hours of daylight?", "It varies by season."),
  ("What season is December?", "Winter in the north."),
  ("What season is June?", "Summer in the north."),
]

def gen_calendar_time():
  items = 
  for q, a in TIME_QA:
  items.append({"category": "calendar_time", "q": q, "a": a})
  for q, a in TIME_QA:
  items.append({"category": "calendar_time",
  "q": "Time: " + q.lower(),
  "a": a})
  write_jsonl("calendar_time", items[:100])

# ---------------------------------------------------------------------------
# weather
# ---------------------------------------------------------------------------

WEATHER_QA = [
  ("What causes rain?", "Water vapor cooling in clouds."),
  ("What causes wind?", "Air moving due to pressure."),
  ("What is a cloud?", "Tiny droplets of water in the sky."),
  ("What is a thunderstorm?", "Storm with thunder and lightning."),
  ("What is a tornado?", "A violent rotating wind column."),
  ("What is a hurricane?", "A huge tropical storm."),
  ("What is a typhoon?", "A hurricane in the Pacific."),
  ("What is a blizzard?", "A heavy snowstorm with wind."),
  ("What is fog?", "A cloud near the ground."),
  ("What is dew?", "Water drops on cool surfaces."),
  ("What is frost?", "Frozen dew on surfaces."),
  ("What is hail?", "Balls of ice from storms."),
  ("What causes lightning?", "Electric build up in clouds."),
  ("What causes thunder?", "Air heated by lightning."),
  ("How hot is the desert?", "Often above 40 C in day."),
  ("How cold is Antarctica?", "Below minus 50 C in winter."),
  ("What is humidity?", "Water vapor in the air."),
  ("What is temperature?", "How hot or cold something is."),
  ("What is Celsius?", "A common temperature scale."),
  ("What is Fahrenheit?", "Another temperature scale."),
  ("What is Kelvin?", "Scientific temperature scale."),
  ("What is the freezing point?", "0 C or 32 F."),
  ("What is body temperature?", "About 37 C or 98.6 F."),
  ("What is sunny?", "Bright with sun, no clouds."),
  ("What is cloudy?", "Sky covered with clouds."),
  ("What is rainy?", "Rain is falling."),
  ("What is snowy?", "Snow is falling."),
  ("What is breezy?", "A light wind blowing."),
  ("What is windy?", "Strong wind blowing."),
  ("Is the weather hot today?", "It can vary, please check."),
  ("Will it rain today?", "I cannot check live weather."),
  ("Is it cold outside?", "I cannot tell without data."),
  ("How is the weather?", "I cannot check current weather."),
  ("What is climate?", "Long term weather patterns."),
  ("What is climate change?", "Long term shift in climate."),
  ("What is global warming?", "Earth warming over time."),
  ("What causes climate change?", "Greenhouse gases mainly."),
  ("What is the greenhouse effect?", "Heat trapped by atmosphere gas."),
  ("What is the ozone layer?", "Layer blocking UV in sky."),
  ("What is acid rain?", "Rain with pollution acids."),
]

def gen_weather():
  items = 
  for q, a in WEATHER_QA:
  items.append({"category": "weather", "q": q, "a": a})
  for q, a in WEATHER_QA:
  items.append({"category": "weather",
  "q": "Weather: " + q.lower(),
  "a": a})
  write_jsonl("weather", items[:80])

# ---------------------------------------------------------------------------
# colors_visual
# ---------------------------------------------------------------------------

COLOR_QA = [
  ("What color do you get mixing red and blue?", "Purple."),
  ("What color do you get mixing yellow and blue?", "Green."),
  ("What color do you get mixing red and yellow?", "Orange."),
  ("What color do you get mixing red and white?", "Pink."),
  ("What color do you get mixing black and white?", "Gray."),
  ("What are primary colors?", "Red, blue, yellow."),
  ("What are secondary colors?", "Green, orange, purple."),
  ("What color is the sky?", "Blue, often."),
  ("What color is grass?", "Green."),
  ("What color is the sun?", "Looks yellow or white."),
  ("What color is fire?", "Red, orange, yellow."),
  ("What color is snow?", "White."),
  ("What color is blood?", "Red."),
  ("What color is the moon?", "White or gray."),
  ("What color is a banana?", "Yellow when ripe."),
  ("What color is a leaf?", "Green, often."),
  ("What color is the ocean?", "Blue, often."),
  ("What color is a stop sign?", "Red."),
  ("What color is a traffic green light?", "Green."),
  ("What color is a school bus?", "Yellow in many places."),
  ("What color is a tomato?", "Red, when ripe."),
  ("What color is a lemon?", "Yellow."),
  ("What color is coal?", "Black."),
  ("What color is a polar bear?", "White."),
  ("What color is night?", "Dark, like black."),
  ("What color is gold?", "Yellow gold."),
  ("What color is silver?", "Shiny gray."),
  ("What is rainbow order?", "Red, orange, yellow, green, blue, indigo, violet."),
  ("What is white made of?", "All colors of light."),
  ("What is black made of?", "Absence of light."),
]

def gen_colors_visual():
  items = 
  for q, a in COLOR_QA:
  items.append({"category": "colors_visual", "q": q, "a": a})
  for q, a in COLOR_QA:
  items.append({"category": "colors_visual",
  "q": "Color: " + q.lower(),
  "a": a})
  write_jsonl("colors_visual", items[:60])

if __name__ == "__main__":
  gen_food_basics()
  gen_food_origins()
  gen_technology_basics()
  gen_technology_programming()
  gen_calendar_time()
  gen_weather()
  gen_colors_visual()
