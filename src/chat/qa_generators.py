"""
Q&A コーパス生成器 (テンプレ展開方式)
事実系のカテゴリは値リスト × テンプレで網羅的に生成する。
"""

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "qa"
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)

def write_jsonl(category: str, items: list):
  path = OUT_DIR / f"{category}.jsonl"
  with path.open('w', encoding='utf-8') as f:
  for it in items:
  f.write(json.dumps(it, ensure_ascii=False) + '\n')
  print(f"[{category}] {len(items)} pairs -> {path}")

# ---------------------------------------------------------------------------
# geography_capitals: 国 → 首都
# ---------------------------------------------------------------------------

CAPITALS = [
  ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
  ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("UK", "London"),
  ("Ireland", "Dublin"), ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
  ("Luxembourg", "Luxembourg City"), ("Switzerland", "Bern"),
  ("Austria", "Vienna"), ("Denmark", "Copenhagen"), ("Sweden", "Stockholm"),
  ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Iceland", "Reykjavik"),
  ("Poland", "Warsaw"), ("Czech Republic", "Prague"), ("Slovakia", "Bratislava"),
  ("Hungary", "Budapest"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
  ("Greece", "Athens"), ("Turkey", "Ankara"), ("Russia", "Moscow"),
  ("Ukraine", "Kyiv"), ("Belarus", "Minsk"), ("Estonia", "Tallinn"),
  ("Latvia", "Riga"), ("Lithuania", "Vilnius"), ("Croatia", "Zagreb"),
  ("Serbia", "Belgrade"), ("Slovenia", "Ljubljana"), ("Albania", "Tirana"),
  ("Japan", "Tokyo"), ("China", "Beijing"), ("South Korea", "Seoul"),
  ("North Korea", "Pyongyang"), ("Mongolia", "Ulaanbaatar"),
  ("India", "New Delhi"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"),
  ("Nepal", "Kathmandu"), ("Sri Lanka", "Colombo"), ("Bhutan", "Thimphu"),
  ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Laos", "Vientiane"),
  ("Cambodia", "Phnom Penh"), ("Myanmar", "Naypyidaw"), ("Malaysia", "Kuala Lumpur"),
  ("Singapore", "Singapore"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
  ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Syria", "Damascus"),
  ("Lebanon", "Beirut"), ("Israel", "Jerusalem"), ("Jordan", "Amman"),
  ("Saudi Arabia", "Riyadh"), ("UAE", "Abu Dhabi"), ("Qatar", "Doha"),
  ("Kuwait", "Kuwait City"), ("Oman", "Muscat"), ("Yemen", "Sanaa"),
  ("Egypt", "Cairo"), ("Libya", "Tripoli"), ("Tunisia", "Tunis"),
  ("Algeria", "Algiers"), ("Morocco", "Rabat"), ("Sudan", "Khartoum"),
  ("Ethiopia", "Addis Ababa"), ("Kenya", "Nairobi"), ("Tanzania", "Dodoma"),
  ("Uganda", "Kampala"), ("Rwanda", "Kigali"), ("Nigeria", "Abuja"),
  ("Ghana", "Accra"), ("Senegal", "Dakar"), ("Mali", "Bamako"),
  ("South Africa", "Pretoria"), ("Zimbabwe", "Harare"), ("Zambia", "Lusaka"),
  ("Mozambique", "Maputo"), ("Angola", "Luanda"), ("Cameroon", "Yaounde"),
  ("USA", "Washington"), ("Canada", "Ottawa"), ("Mexico", "Mexico City"),
  ("Cuba", "Havana"), ("Jamaica", "Kingston"), ("Haiti", "Port-au-Prince"),
  ("Guatemala", "Guatemala City"), ("Honduras", "Tegucigalpa"),
  ("Nicaragua", "Managua"), ("Costa Rica", "San Jose"), ("Panama", "Panama City"),
  ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
  ("Peru", "Lima"), ("Bolivia", "La Paz"), ("Colombia", "Bogota"),
  ("Venezuela", "Caracas"), ("Ecuador", "Quito"), ("Uruguay", "Montevideo"),
  ("Paraguay", "Asuncion"), ("Australia", "Canberra"),
  ("New Zealand", "Wellington"), ("Fiji", "Suva"), ("Samoa", "Apia"),
  ("Tonga", "Nukualofa"), ("Papua New Guinea", "Port Moresby"),
]

CAPITAL_QFORMS = [
  "What is the capital of {country}?",
  "What's the capital of {country}?",
  "Tell me the capital of {country}.",
  "Capital of {country}?",
  "Which city is the capital of {country}?",
  "Name the capital of {country}.",
  "Do you know the capital of {country}?",
  "What city is {country}'s capital?",
]
CAPITAL_AFORMS = [
  "The capital of {country} is {capital}.",
  "It is {capital}.",
  "{capital}.",
  "{capital}, the capital of {country}.",
  "{country}'s capital is {capital}.",
]

def gen_capitals():
  items = 
  for country, capital in CAPITALS:
  q = random.choice(CAPITAL_QFORMS).format(country=country)
  a = random.choice(CAPITAL_AFORMS).format(country=country, capital=capital)
  items.append({"category": "geography_capitals", "q": q, "a": a})
  # 一部の有名国は別フォームでも追加して 200件目標
  extras = [
  ("France", "Paris"), ("Japan", "Tokyo"), ("USA", "Washington"),
  ("UK", "London"), ("Germany", "Berlin"), ("Italy", "Rome"),
  ("Spain", "Madrid"), ("China", "Beijing"), ("India", "New Delhi"),
  ("Brazil", "Brasilia"), ("Russia", "Moscow"), ("Canada", "Ottawa"),
  ("Australia", "Canberra"), ("Mexico", "Mexico City"),
  ("Egypt", "Cairo"), ("Greece", "Athens"), ("Turkey", "Ankara"),
  ("Argentina", "Buenos Aires"), ("South Korea", "Seoul"),
  ("Indonesia", "Jakarta"), ("Thailand", "Bangkok"),
  ]
  # 80件 unique target でも CAPITALS が 110+ あるので extras 不要そうだが念のため
  for country, capital in extras * 3:
  if len(items) >= 220:
  break
  q = random.choice(CAPITAL_QFORMS).format(country=country)
  a = random.choice(CAPITAL_AFORMS).format(country=country, capital=capital)
  items.append({"category": "geography_capitals", "q": q, "a": a})
  write_jsonl("geography_capitals", items[:200])

# ---------------------------------------------------------------------------
# geography_continents: 国 → 大陸
# ---------------------------------------------------------------------------

CONTINENT_OF = {
  "Europe": ["France", "Germany", "Italy", "Spain", "Portugal", "UK", "Ireland",
  "Netherlands", "Belgium", "Switzerland", "Austria", "Denmark",
  "Sweden", "Norway", "Finland", "Iceland", "Poland", "Greece"],
  "Asia": ["Japan", "China", "South Korea", "India", "Pakistan", "Thailand",
  "Vietnam", "Indonesia", "Philippines", "Malaysia", "Singapore",
  "Iran", "Iraq", "Saudi Arabia", "Israel", "Turkey", "Mongolia",
  "Nepal", "Sri Lanka", "Bangladesh"],
  "Africa": ["Egypt", "Nigeria", "Kenya", "Ethiopia", "South Africa",
  "Morocco", "Algeria", "Ghana", "Tanzania", "Uganda", "Sudan",
  "Libya", "Tunisia", "Zimbabwe", "Angola"],
  "North America": ["USA", "Canada", "Mexico", "Cuba", "Jamaica", "Haiti",
  "Guatemala", "Honduras", "Costa Rica", "Panama"],
  "South America": ["Brazil", "Argentina", "Chile", "Peru", "Bolivia",
  "Colombia", "Venezuela", "Ecuador", "Uruguay", "Paraguay"],
  "Oceania": ["Australia", "New Zealand", "Fiji", "Samoa", "Tonga",
  "Papua New Guinea"],
}

CONT_QFORMS = [
  "Which continent is {country} in?",
  "What continent does {country} belong to?",
  "{country} is in which continent?",
  "Tell me the continent of {country}.",
  "Where in the world is {country}?",
]
CONT_AFORMS = [
  "{country} is in {continent}.",
  "It is in {continent}.",
  "{continent}.",
  "{country} belongs to {continent}.",
]

def gen_continents():
  items = 
  for cont, countries in CONTINENT_OF.items():
  for country in countries:
  q = random.choice(CONT_QFORMS).format(country=country)
  a = random.choice(CONT_AFORMS).format(country=country, continent=cont)
  items.append({"category": "geography_continents", "q": q, "a": a})
  write_jsonl("geography_continents", items[:100])

# ---------------------------------------------------------------------------
# geography_rivers
# ---------------------------------------------------------------------------

RIVERS = [
  ("Nile", "Africa", "Egypt and Sudan"),
  ("Amazon", "South America", "Brazil and Peru"),
  ("Yangtze", "Asia", "China"),
  ("Mississippi", "North America", "the United States"),
  ("Yenisei", "Asia", "Russia"),
  ("Yellow River", "Asia", "China"),
  ("Ob", "Asia", "Russia"),
  ("Parana", "South America", "Argentina and Brazil"),
  ("Congo", "Africa", "Central Africa"),
  ("Amur", "Asia", "Russia and China"),
  ("Lena", "Asia", "Russia"),
  ("Mekong", "Asia", "Southeast Asia"),
  ("Niger", "Africa", "West Africa"),
  ("Murray", "Oceania", "Australia"),
  ("Volga", "Europe", "Russia"),
  ("Indus", "Asia", "Pakistan"),
  ("Ganges", "Asia", "India"),
  ("Brahmaputra", "Asia", "India and Bangladesh"),
  ("Danube", "Europe", "Central and Eastern Europe"),
  ("Rhine", "Europe", "Germany and the Netherlands"),
  ("Seine", "Europe", "France"),
  ("Thames", "Europe", "England"),
  ("Tiber", "Europe", "Italy"),
  ("Po", "Europe", "Italy"),
  ("Tagus", "Europe", "Spain and Portugal"),
  ("Loire", "Europe", "France"),
  ("Rio Grande", "North America", "USA and Mexico"),
  ("Colorado", "North America", "the United States"),
  ("Columbia", "North America", "USA and Canada"),
  ("Yukon", "North America", "USA and Canada"),
  ("Saint Lawrence", "North America", "Canada"),
  ("Orinoco", "South America", "Venezuela"),
  ("Tigris", "Asia", "Iraq"),
  ("Euphrates", "Asia", "Iraq and Syria"),
  ("Jordan", "Asia", "Israel and Jordan"),
  ("Zambezi", "Africa", "Southern Africa"),
  ("Limpopo", "Africa", "Southern Africa"),
  ("Orange", "Africa", "South Africa"),
]

RIVER_QFORMS = [
  "Where does the {river} river flow?",
  "Which country has the {river} river?",
  "Where is the {river}?",
  "Tell me about the {river} river.",
  "Where can I find the {river}?",
]
RIVER_AFORMS = [
  "The {river} flows through {place}.",
  "The {river} is in {place}.",
  "It flows through {place}.",
  "{place}.",
]

def gen_rivers():
  items = 
  for river, _continent, place in RIVERS:
  q = random.choice(RIVER_QFORMS).format(river=river)
  a = random.choice(RIVER_AFORMS).format(river=river, place=place)
  items.append({"category": "geography_rivers", "q": q, "a": a})
  # 80件目標、現在 38件 → 倍にする（別表現）
  extras = 
  for river, _continent, place in RIVERS:
  q = f"The {river} river is located in?"
  a = f"It is in {place}."
  extras.append({"category": "geography_rivers", "q": q, "a": a})
  items.extend(extras)
  write_jsonl("geography_rivers", items[:80])

# ---------------------------------------------------------------------------
# geography_oceans
# ---------------------------------------------------------------------------

OCEANS_FACTS = [
  ("What is the largest ocean?", "The Pacific Ocean is the largest."),
  ("What is the smallest ocean?", "The Arctic Ocean is the smallest."),
  ("How many oceans are there?", "There are five oceans."),
  ("Name the five oceans.", "Pacific, Atlantic, Indian, Arctic, Southern."),
  ("Which ocean is between America and Europe?", "The Atlantic Ocean."),
  ("Which ocean is between Asia and America?", "The Pacific Ocean."),
  ("Which ocean is south of India?", "The Indian Ocean."),
  ("Which ocean is around Antarctica?", "The Southern Ocean."),
  ("Which ocean is near the North Pole?", "The Arctic Ocean."),
  ("Which ocean is the deepest?", "The Pacific Ocean is deepest."),
  ("Where is the Mariana Trench?", "In the Pacific Ocean."),
  ("Is the Pacific bigger than the Atlantic?", "Yes, much bigger."),
  ("Is the Atlantic warmer than the Arctic?", "Yes, it is warmer."),
  ("What ocean is east of Africa?", "The Indian Ocean."),
  ("What ocean is west of Africa?", "The Atlantic Ocean."),
  ("Where is the Bermuda Triangle?", "In the Atlantic Ocean."),
  ("What is the second largest ocean?", "The Atlantic Ocean."),
  ("Which ocean surrounds Australia?", "Pacific and Indian Oceans."),
  ("Where do the Gulf Stream waters flow?", "Across the North Atlantic."),
  ("Is the Indian Ocean warm?", "Yes, mostly warm waters."),
]

def gen_oceans():
  items = 
  for q, a in OCEANS_FACTS:
  items.append({"category": "geography_oceans", "q": q, "a": a})
  # 20件 → 60件目標、3倍に膨らます (q phrasing 変える)
  for q, a in OCEANS_FACTS:
  q2 = q.replace("What is", "Can you tell me what is")
  items.append({"category": "geography_oceans", "q": q2, "a": a})
  for q, a in OCEANS_FACTS:
  q3 = "Quick question: " + q.lower()
  items.append({"category": "geography_oceans", "q": q3, "a": a})
  write_jsonl("geography_oceans", items[:60])

# ---------------------------------------------------------------------------
# math_arithmetic: a + b, a - b, etc.
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
  0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
  5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
  10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
  15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
  20: "twenty",
}

def n_to_word(n):
  if n in NUMBER_WORDS:
  return NUMBER_WORDS[n]
  return str(n)

ADD_QFORMS = [
  "What is {a} plus {b}?",
  "{a} plus {b}?",
  "What is {a} + {b}?",
  "How much is {a} plus {b}?",
  "Tell me {a} plus {b}.",
  "Calculate {a} + {b}.",
]
ADD_AFORMS = [
  "It is {c}.",
  "{c}.",
  "The answer is {c}.",
  "{a} plus {b} is {c}.",
  "Equals {c}.",
]

def gen_math_arithmetic():
  items = 
  # 加算 100件
  pairs = set()
  while len(pairs) < 100:
  a = random.randint(1, 50)
  b = random.randint(1, 50)
  pairs.add((a, b))
  for a, b in pairs:
  c = a + b
  q = random.choice(ADD_QFORMS).format(a=a, b=b)
  ans = random.choice(ADD_AFORMS).format(a=a, b=b, c=c)
  items.append({"category": "math_arithmetic", "q": q, "a": ans})
  # 減算 100件
  pairs = set()
  while len(pairs) < 100:
  a = random.randint(5, 50)
  b = random.randint(1, a)
  pairs.add((a, b))
  sub_q = ["What is {a} minus {b}?", "{a} minus {b}?", "What is {a} - {b}?",
  "How much is {a} minus {b}?"]
  sub_a = ["It is {c}.", "{c}.", "The answer is {c}.",
  "{a} minus {b} is {c}."]
  for a, b in pairs:
  c = a - b
  q = random.choice(sub_q).format(a=a, b=b)
  ans = random.choice(sub_a).format(a=a, b=b, c=c)
  items.append({"category": "math_arithmetic", "q": q, "a": ans})
  # 簡単な乗算 / 除算 100件
  pairs = set()
  while len(pairs) < 50:
  a = random.randint(1, 12)
  b = random.randint(1, 12)
  pairs.add((a, b))
  mul_q = ["What is {a} times {b}?", "{a} times {b}?",
  "What is {a} multiplied by {b}?"]
  mul_a = ["It is {c}.", "{c}.", "The answer is {c}."]
  for a, b in pairs:
  c = a * b
  q = random.choice(mul_q).format(a=a, b=b)
  ans = random.choice(mul_a).format(a=a, b=b, c=c)
  items.append({"category": "math_arithmetic", "q": q, "a": ans})
  # 除算
  pairs = set()
  while len(pairs) < 50:
  c = random.randint(1, 12)
  b = random.randint(1, 12)
  a = c * b
  pairs.add((a, b, c))
  div_q = ["What is {a} divided by {b}?", "{a} divided by {b}?",
  "What is {a} / {b}?"]
  div_a = ["It is {c}.", "{c}.", "The answer is {c}."]
  for a, b, c in pairs:
  q = random.choice(div_q).format(a=a, b=b)
  ans = random.choice(div_a).format(a=a, b=b, c=c)
  items.append({"category": "math_arithmetic", "q": q, "a": ans})
  random.shuffle(items)
  write_jsonl("math_arithmetic", items[:300])

# ---------------------------------------------------------------------------
# math_multiplication (九九)
# ---------------------------------------------------------------------------

def gen_math_multiplication():
  items = 
  # 1..12 x 1..12 = 144 件、150件目標は phrasing 変えて補完
  for a in range(2, 13):
  for b in range(2, 13):
  c = a * b
  q = random.choice([
  f"{a} times {b}?",
  f"What is {a} times {b}?",
  f"{a} x {b}?",
  f"What's {a} * {b}?",
  ])
  ans = random.choice([f"{c}.", f"It is {c}.", f"That is {c}."])
  items.append({"category": "math_multiplication", "q": q, "a": ans})
  write_jsonl("math_multiplication", items[:150])

# ---------------------------------------------------------------------------
# math_definitions
# ---------------------------------------------------------------------------

MATH_DEFS = [
  ("What is a prime number?", "A number divisible only by 1 and itself."),
  ("What is an even number?", "A number divisible by 2."),
  ("What is an odd number?", "A number not divisible by 2."),
  ("What is a square number?", "A number that is some n times n."),
  ("What is a fraction?", "A part of a whole, like 1/2."),
  ("What is a decimal?", "A number with a point, like 3.14."),
  ("What is pi?", "Pi is about 3.14159, a circle ratio."),
  ("What is zero?", "Zero means nothing, the absence of count."),
  ("What is infinity?", "Infinity is endless, without bound."),
  ("What is a triangle?", "A shape with three sides."),
  ("What is a square shape?", "A shape with four equal sides."),
  ("What is a circle?", "A round shape with no corners."),
  ("What is a rectangle?", "A shape with four right angles."),
  ("What is a sum?", "The result of adding numbers."),
  ("What is a difference?", "The result of subtracting numbers."),
  ("What is a product?", "The result of multiplying numbers."),
  ("What is a quotient?", "The result of dividing numbers."),
  ("What is a variable?", "A symbol for an unknown number."),
  ("What is an equation?", "Two sides set equal by an equals sign."),
  ("What is a function?", "A rule that maps inputs to outputs."),
  ("What is geometry?", "The math of shapes and space."),
  ("What is algebra?", "Math with letters and equations."),
  ("What is calculus?", "Math of change and slopes."),
  ("What is a percent?", "A part out of one hundred."),
  ("What is an angle?", "A turn between two lines."),
  ("What is a right angle?", "An angle of ninety degrees."),
  ("What is a parallel?", "Two lines never meeting."),
  ("What is a polygon?", "A shape with many straight sides."),
  ("What is symmetry?", "A balance across an axis."),
  ("What is a graph?", "A drawing of data or functions."),
  ("What does sum mean?", "It means to add up numbers."),
  ("What does product mean?", "It means to multiply numbers."),
  ("What is a remainder?", "What is left after division."),
  ("What is a divisor?", "The number you divide by."),
  ("What is a factor?", "A number that divides another."),
  ("What is multiplication?", "Repeated addition of a number."),
  ("What is division?", "Splitting a number into equal parts."),
  ("What is subtraction?", "Taking one number from another."),
  ("What is addition?", "Putting numbers together."),
  ("What is an integer?", "A whole number, positive or negative."),
]

def gen_math_definitions():
  items = 
  for q, a in MATH_DEFS:
  items.append({"category": "math_definitions", "q": q, "a": a})
  # 40件 → 80件 phrasing 変えて倍にする
  for q, a in MATH_DEFS:
  if q.startswith("What is"):
  q2 = "Define " + q.replace("What is ", "").replace("?", "") + "."
  items.append({"category": "math_definitions", "q": q2, "a": a})
  write_jsonl("math_definitions", items[:80])

if __name__ == "__main__":
  gen_capitals()
  gen_continents()
  gen_rivers()
  gen_oceans()
  gen_math_arithmetic()
  gen_math_multiplication()
  gen_math_definitions()
  print("\n[done] geography_* + math_* (合計 ~900件) を生成")
