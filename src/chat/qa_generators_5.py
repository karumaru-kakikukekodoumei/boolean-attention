"""
Q&A 生成器 5: 残り全カテゴリ
sports_facts, music_basics, art_basics, currency, languages_spoken,
occupations, common_objects, transportation, emotions_feelings,
opinions_simple, yes_no_facts, comparisons, philosophy_simple,
shakespeare_dialogue, casual_chitchat, self_identity, follow_up_short
"""

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "qa"
OUT_DIR.mkdir(exist_ok=True)

random.seed(46)

def write_jsonl(category: str, items: list):
  path = OUT_DIR / f"{category}.jsonl"
  with path.open('w', encoding='utf-8') as f:
  for it in items:
  f.write(json.dumps(it, ensure_ascii=False) + '\n')
  print(f"[{category}] {len(items)} pairs -> {path}")

# ---------------------------------------------------------------------------

SPORTS_QA = [
  ("How many players on a soccer team?", "Eleven players."),
  ("How many players on a basketball team?", "Five players on court."),
  ("How many players on a baseball team?", "Nine players on field."),
  ("How many players on a volleyball team?", "Six players."),
  ("How many players on a hockey team?", "Six players on ice."),
  ("How many players on a rugby team?", "15 in rugby union."),
  ("How long is a soccer match?", "90 minutes."),
  ("How long is a basketball game?", "About 48 minutes pro."),
  ("How long is a tennis set?", "Variable, often around hours."),
  ("How many sets in tennis?", "Best of 3 or 5."),
  ("How often is the World Cup?", "Every 4 years."),
  ("How often is the Olympics?", "Every 4 years."),
  ("Where were Olympics born?", "In ancient Greece."),
  ("When were modern Olympics started?", "In 1896."),
  ("Who is Pele?", "Brazilian soccer legend."),
  ("Who is Maradona?", "Argentine soccer legend."),
  ("Who is Messi?", "Argentine soccer star."),
  ("Who is Ronaldo?", "Portuguese soccer star."),
  ("Who is Michael Jordan?", "NBA basketball legend."),
  ("Who is LeBron James?", "NBA basketball star."),
  ("Who is Kobe Bryant?", "Late NBA basketball star."),
  ("Who is Tiger Woods?", "Golf champion."),
  ("Who is Serena Williams?", "Tennis champion."),
  ("Who is Roger Federer?", "Swiss tennis legend."),
  ("Who is Rafael Nadal?", "Spanish tennis legend."),
  ("Who is Novak Djokovic?", "Serbian tennis star."),
  ("Who is Usain Bolt?", "Sprinter of Jamaica."),
  ("Who is Muhammad Ali?", "Legendary boxer."),
  ("What sport uses a bat?", "Baseball uses a bat."),
  ("What sport uses a racket?", "Tennis, badminton, more."),
  ("What sport uses a puck?", "Ice hockey."),
  ("What is FIFA?", "World soccer governing body."),
  ("What is NBA?", "US basketball league."),
  ("What is MLB?", "US baseball league."),
  ("What is NFL?", "US football league."),
  ("What is NHL?", "Ice hockey league."),
  ("What is Wimbledon?", "Famous tennis tournament."),
  ("What is the Super Bowl?", "NFL championship game."),
  ("What is the World Series?", "MLB championship."),
  ("How many strikes are an out?", "Three strikes."),
  ("How many balls walk a batter?", "Four balls."),
  ("How many innings in baseball?", "Nine innings."),
  ("How many quarters in basketball?", "Four quarters."),
  ("How many halves in soccer?", "Two halves."),
  ("How tall is a basketball hoop?", "Ten feet high."),
  ("How long is a marathon?", "About 42 km."),
  ("How fast can humans run?", "Up to about 44 km/h."),
  ("What is a hat-trick?", "Three goals by one player."),
  ("What is offside?", "An illegal forward position."),
  ("What is a penalty kick?", "A free kick from 11 meters."),
  ("What is a free kick?", "A kick after a foul."),
  ("What is a yellow card?", "A warning in soccer."),
  ("What is a red card?", "A send-off in soccer."),
  ("What is a slam dunk?", "A forceful basketball score."),
  ("What is a three-pointer?", "A long basketball shot."),
  ("What is a home run?", "Hitting ball over the fence."),
  ("What is a touchdown?", "A scoring play in football."),
  ("What is a goal?", "A score in soccer or hockey."),
  ("What is a knockout?", "A boxer down for 10 seconds."),
  ("What is an ace in tennis?", "An unreturnable serve."),
]

def gen_sports_facts():
  items = 
  for q, a in SPORTS_QA:
  items.append({"category": "sports_facts", "q": q, "a": a})
  for q, a in SPORTS_QA:
  items.append({"category": "sports_facts",
  "q": "Sports: " + q.lower(),
  "a": a})
  write_jsonl("sports_facts", items[:120])

# ---------------------------------------------------------------------------

MUSIC_QA = [
  ("How many strings on a guitar?", "Usually six."),
  ("How many strings on a violin?", "Four strings."),
  ("How many strings on a cello?", "Four strings."),
  ("How many strings on a bass guitar?", "Usually four."),
  ("How many keys on a piano?", "88 keys."),
  ("How many white keys on a piano?", "52 white keys."),
  ("How many black keys on a piano?", "36 black keys."),
  ("Name a wind instrument.", "Flute, trumpet, clarinet."),
  ("Name a string instrument.", "Violin, guitar, cello."),
  ("Name a percussion instrument.", "Drum, cymbal, xylophone."),
  ("Who is Mozart?", "Austrian classical composer."),
  ("Who is Beethoven?", "German classical composer."),
  ("Who is Bach?", "German Baroque composer."),
  ("Who is The Beatles?", "Famous British rock band."),
  ("Who is Elvis Presley?", "King of rock and roll."),
  ("Who is Michael Jackson?", "King of pop."),
  ("Who is Madonna?", "Queen of pop."),
  ("Who is Bob Dylan?", "Folk and rock singer-songwriter."),
  ("Who is Frank Sinatra?", "American crooner."),
  ("Who is Freddie Mercury?", "Queen's lead vocalist."),
  ("What is Queen?", "Famous British rock band."),
  ("What is The Beatles?", "Famous British rock band."),
  ("What is jazz?", "An African-American music style."),
  ("What is rock?", "A loud guitar-driven music style."),
  ("What is classical music?", "Music in formal European tradition."),
  ("What is hip hop?", "Music with rap and beats."),
  ("What is pop music?", "Popular mainstream music."),
  ("What is country music?", "American rural music style."),
  ("What is reggae?", "Music from Jamaica."),
  ("What is blues?", "Sad emotional African-American music."),
  ("Who wrote Beethoven's 9th?", "Beethoven himself."),
  ("Who wrote Yesterday?", "The Beatles, Paul McCartney."),
  ("Who wrote Bohemian Rhapsody?", "Queen, Freddie Mercury."),
  ("What is a song?", "Music with words to sing."),
  ("What is a chord?", "Multiple notes played together."),
  ("What is a scale?", "A sequence of notes."),
  ("What is rhythm?", "The beat pattern of music."),
  ("What is melody?", "The main tune of a song."),
  ("What is harmony?", "Notes that sound together."),
  ("What is tempo?", "How fast music plays."),
]

def gen_music_basics():
  items = 
  for q, a in MUSIC_QA:
  items.append({"category": "music_basics", "q": q, "a": a})
  for q, a in MUSIC_QA:
  items.append({"category": "music_basics",
  "q": "Music: " + q.lower(),
  "a": a})
  write_jsonl("music_basics", items[:80])

# ---------------------------------------------------------------------------

ART_QA = [
  ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
  ("Who painted The Starry Night?", "Vincent van Gogh."),
  ("Who painted Sunflowers?", "Vincent van Gogh."),
  ("Who painted Guernica?", "Pablo Picasso."),
  ("Who painted The Scream?", "Edvard Munch."),
  ("Who painted The Last Supper?", "Leonardo da Vinci."),
  ("Who sculpted David?", "Michelangelo."),
  ("Who painted the Sistine Chapel?", "Michelangelo."),
  ("Who painted Girl with a Pearl Earring?", "Johannes Vermeer."),
  ("Where is the Mona Lisa?", "In the Louvre, Paris."),
  ("Where is the Sistine Chapel?", "In Vatican City."),
  ("What is Cubism?", "An art style of geometric forms."),
  ("Who started Cubism?", "Picasso and Braque."),
  ("What is Impressionism?", "Art style of light and color."),
  ("Name an Impressionist painter.", "Claude Monet."),
  ("What is Surrealism?", "Dream-like, illogical art."),
  ("Who is Salvador Dali?", "Spanish Surrealist painter."),
  ("What is Renaissance art?", "Realistic art of 14th to 17th c."),
  ("What is Baroque art?", "Dramatic detailed 17th c art."),
  ("What is sculpture?", "Three-dimensional art."),
  ("What is a portrait?", "A painting of a person."),
  ("What is a landscape?", "A painting of scenery."),
  ("What is a still life?", "A painting of objects."),
  ("What is a self-portrait?", "Artist painting themselves."),
  ("What is watercolor?", "Paint mixed with water."),
  ("What is oil paint?", "Pigment in oil."),
  ("What is acrylic paint?", "Fast-drying water-based paint."),
  ("What is a museum?", "A place that displays art."),
  ("Name a famous museum.", "The Louvre in Paris."),
  ("Name an art museum in NY.", "The MoMA."),
]

def gen_art_basics():
  items = 
  for q, a in ART_QA:
  items.append({"category": "art_basics", "q": q, "a": a})
  for q, a in ART_QA:
  items.append({"category": "art_basics",
  "q": "Art: " + q.lower(),
  "a": a})
  write_jsonl("art_basics", items[:60])

# ---------------------------------------------------------------------------

CURRENCY_QA = [
  ("What is the currency of Japan?", "The Japanese yen."),
  ("What is the currency of the USA?", "The US dollar."),
  ("What is the currency of the UK?", "The British pound."),
  ("What is the currency of France?", "The euro."),
  ("What is the currency of Germany?", "The euro."),
  ("What is the currency of China?", "The Chinese yuan."),
  ("What is the currency of India?", "The Indian rupee."),
  ("What is the currency of Brazil?", "The Brazilian real."),
  ("What is the currency of Russia?", "The Russian ruble."),
  ("What is the currency of Mexico?", "The Mexican peso."),
  ("What is the currency of Canada?", "The Canadian dollar."),
  ("What is the currency of Australia?", "The Australian dollar."),
  ("What is the currency of Switzerland?", "The Swiss franc."),
  ("What is the currency of Sweden?", "The Swedish krona."),
  ("What is the currency of Norway?", "The Norwegian krone."),
  ("What is the currency of Denmark?", "The Danish krone."),
  ("What is the currency of Korea?", "The Korean won."),
  ("What is the currency of Turkey?", "The Turkish lira."),
  ("What is the currency of Egypt?", "The Egyptian pound."),
  ("What is the currency of South Africa?", "The rand."),
  ("What is the currency of Thailand?", "The Thai baht."),
  ("What is the currency of Vietnam?", "The Vietnamese dong."),
  ("What is the currency of Indonesia?", "The Indonesian rupiah."),
  ("What is the currency of Singapore?", "The Singapore dollar."),
  ("What is the currency of Malaysia?", "The Malaysian ringgit."),
  ("What is the currency of Philippines?", "The Philippine peso."),
  ("What is the currency of Argentina?", "The Argentine peso."),
  ("What is the currency of Italy?", "The euro."),
  ("What is the currency of Spain?", "The euro."),
  ("What is the currency of the Eurozone?", "The euro."),
  ("What is BTC?", "Bitcoin, a cryptocurrency."),
  ("What is ETH?", "Ethereum, a cryptocurrency."),
  ("What is fiat money?", "Money declared by a government."),
  ("What is cash?", "Physical money, paper or coin."),
  ("What is a coin?", "Small metal piece of money."),
  ("What is a banknote?", "Paper money."),
  ("What is inflation?", "Prices going up over time."),
  ("What is deflation?", "Prices going down over time."),
  ("What is exchange rate?", "Value of one currency vs another."),
]

def gen_currency():
  items = 
  for q, a in CURRENCY_QA:
  items.append({"category": "currency", "q": q, "a": a})
  for q, a in CURRENCY_QA:
  items.append({"category": "currency",
  "q": "Currency: " + q.lower(),
  "a": a})
  write_jsonl("currency", items[:80])

# ---------------------------------------------------------------------------

LANG_SPOKEN_QA = [
  ("What language is spoken in Japan?", "Japanese."),
  ("What language is spoken in France?", "French."),
  ("What language is spoken in Germany?", "German."),
  ("What language is spoken in Italy?", "Italian."),
  ("What language is spoken in Spain?", "Spanish."),
  ("What language is spoken in Portugal?", "Portuguese."),
  ("What language is spoken in Brazil?", "Portuguese."),
  ("What language is spoken in China?", "Mandarin Chinese."),
  ("What language is spoken in Korea?", "Korean."),
  ("What language is spoken in Russia?", "Russian."),
  ("What language is spoken in Greece?", "Greek."),
  ("What language is spoken in Turkey?", "Turkish."),
  ("What language is spoken in Egypt?", "Arabic."),
  ("What language is spoken in Saudi Arabia?", "Arabic."),
  ("What language is spoken in Mexico?", "Spanish."),
  ("What language is spoken in Argentina?", "Spanish."),
  ("What language is spoken in Sweden?", "Swedish."),
  ("What language is spoken in Norway?", "Norwegian."),
  ("What language is spoken in Denmark?", "Danish."),
  ("What language is spoken in Finland?", "Finnish."),
  ("What language is spoken in Poland?", "Polish."),
  ("What language is spoken in Vietnam?", "Vietnamese."),
  ("What language is spoken in Thailand?", "Thai."),
  ("What language is spoken in Indonesia?", "Indonesian."),
  ("What language is spoken in Philippines?", "Filipino and English."),
  ("What is the most spoken language?", "Mandarin or English."),
  ("How many languages exist?", "About 7000 living languages."),
  ("What is Latin?", "An ancient Roman language."),
  ("Is Latin still spoken?", "Mostly only as a dead language."),
  ("What is Esperanto?", "A constructed universal language."),
]

def gen_languages_spoken():
  items = 
  for q, a in LANG_SPOKEN_QA:
  items.append({"category": "languages_spoken", "q": q, "a": a})
  for q, a in LANG_SPOKEN_QA:
  items.append({"category": "languages_spoken",
  "q": "Language: " + q.lower(),
  "a": a})
  write_jsonl("languages_spoken", items[:60])

# ---------------------------------------------------------------------------

OCCUPATIONS_QA = [
  ("What does a chef do?", "A chef cooks food."),
  ("What does a doctor do?", "A doctor treats the sick."),
  ("What does a nurse do?", "A nurse cares for patients."),
  ("What does a teacher do?", "A teacher educates students."),
  ("What does a farmer do?", "A farmer grows food."),
  ("What does a baker do?", "A baker makes bread."),
  ("What does a butcher do?", "A butcher cuts meat."),
  ("What does a fisherman do?", "Catches fish for a living."),
  ("What does a carpenter do?", "Builds with wood."),
  ("What does a plumber do?", "Fixes pipes and water."),
  ("What does an electrician do?", "Works with electrical systems."),
  ("What does a mechanic do?", "Repairs cars and machines."),
  ("What does a driver do?", "Drives a vehicle for work."),
  ("What does a pilot do?", "Flies airplanes."),
  ("What does a sailor do?", "Works on a ship."),
  ("What does a soldier do?", "Serves in the military."),
  ("What does a police officer do?", "Keeps order, enforces law."),
  ("What does a firefighter do?", "Fights fires, saves lives."),
  ("What does a paramedic do?", "Gives first aid emergencies."),
  ("What does a lawyer do?", "Practices law in court."),
  ("What does a judge do?", "Decides cases in court."),
  ("What does an accountant do?", "Manages money records."),
  ("What does a banker do?", "Works with money in a bank."),
  ("What does an engineer do?", "Designs and builds things."),
  ("What does a programmer do?", "Writes computer software."),
  ("What does a scientist do?", "Studies the natural world."),
  ("What does a researcher do?", "Investigates a topic."),
  ("What does a writer do?", "Writes stories or articles."),
  ("What does a journalist do?", "Reports news stories."),
  ("What does an artist do?", "Creates visual art."),
  ("What does a musician do?", "Plays or makes music."),
  ("What does a singer do?", "Sings songs."),
  ("What does an actor do?", "Performs in plays or films."),
  ("What does a dancer do?", "Performs dance art."),
  ("What does a photographer do?", "Takes photos for a living."),
  ("What does a designer do?", "Designs products or visuals."),
  ("What does an architect do?", "Designs buildings."),
  ("What does a barber do?", "Cuts hair."),
  ("What does a tailor do?", "Sews clothes by hand."),
  ("What does a librarian do?", "Manages a library."),
  ("What does a janitor do?", "Cleans buildings."),
  ("What does a waiter do?", "Serves food at a restaurant."),
  ("What does a cashier do?", "Handles money at a register."),
  ("What does a manager do?", "Oversees a team or store."),
  ("What does a CEO do?", "Leads a whole company."),
  ("What does a coach do?", "Trains athletes or teams."),
  ("What does a referee do?", "Officiates a sports match."),
  ("What does an astronaut do?", "Travels to outer space."),
  ("What does a veterinarian do?", "Treats sick animals."),
  ("What does a dentist do?", "Cares for teeth."),
]

def gen_occupations():
  items = 
  for q, a in OCCUPATIONS_QA:
  items.append({"category": "occupations", "q": q, "a": a})
  for q, a in OCCUPATIONS_QA:
  items.append({"category": "occupations",
  "q": "Job: " + q.lower(),
  "a": a})
  write_jsonl("occupations", items[:100])

# ---------------------------------------------------------------------------

OBJECT_QA = [
  ("What is a chair used for?", "Sitting on it."),
  ("What is a table used for?", "Holding items or eating."),
  ("What is a bed used for?", "Sleeping in."),
  ("What is a lamp used for?", "Giving off light."),
  ("What is a clock used for?", "Showing the time."),
  ("What is a phone used for?", "Calling and messaging."),
  ("What is a computer used for?", "Many tasks: work, fun, etc."),
  ("What is a book used for?", "Reading and learning."),
  ("What is a pen used for?", "Writing with ink."),
  ("What is a pencil used for?", "Writing with graphite."),
  ("What is a knife used for?", "Cutting things."),
  ("What is a spoon used for?", "Eating liquid food."),
  ("What is a fork used for?", "Eating solid food."),
  ("What is a plate used for?", "Holding food when eating."),
  ("What is a cup used for?", "Drinking liquids."),
  ("What is a bottle used for?", "Holding liquids."),
  ("What is a key used for?", "Opening locks."),
  ("What is a lock used for?", "Keeping things secure."),
  ("What is a door used for?", "Entering or leaving rooms."),
  ("What is a window used for?", "Letting in light and air."),
  ("What is a mirror used for?", "Seeing your own image."),
  ("What is an umbrella used for?", "Protection from rain."),
  ("What is a coat used for?", "Keeping warm outside."),
  ("What is shoes for?", "Protecting your feet."),
  ("What is a hat for?", "Covering your head."),
  ("What is glasses for?", "Helping you see clearly."),
  ("What is soap used for?", "Cleaning skin."),
  ("What is a towel used for?", "Drying after washing."),
  ("What is a broom used for?", "Sweeping floors."),
  ("What is a hammer used for?", "Driving nails."),
  ("What is a screwdriver used for?", "Turning screws."),
  ("What is a saw used for?", "Cutting wood."),
  ("What is a paintbrush used for?", "Applying paint."),
  ("What is scissors used for?", "Cutting paper or fabric."),
  ("What is glue used for?", "Sticking things together."),
  ("What is tape used for?", "Sticking or sealing."),
  ("What is a watch used for?", "Telling time on the wrist."),
  ("What is a wallet used for?", "Holding cash and cards."),
  ("What is a bag used for?", "Carrying items."),
  ("What is a backpack used for?", "Carrying on your back."),
  ("What is a fridge used for?", "Keeping food cold."),
  ("What is a microwave used for?", "Heating food quickly."),
  ("What is an oven used for?", "Baking and roasting."),
  ("What is a stove used for?", "Cooking on burners."),
  ("What is a kettle used for?", "Boiling water."),
  ("What is a toaster used for?", "Toasting bread."),
  ("What is a washing machine for?", "Washing clothes."),
  ("What is a dryer used for?", "Drying clothes."),
  ("What is a vacuum used for?", "Cleaning carpets and floors."),
  ("What is a TV used for?", "Watching shows and films."),
  ("What is a radio used for?", "Listening to broadcasts."),
  ("What is a camera used for?", "Taking photos or video."),
  ("What is a piano used for?", "Playing music."),
  ("What is a ladder used for?", "Climbing high places."),
  ("What is a bucket used for?", "Holding water or items."),
  ("What is a needle used for?", "Sewing fabric."),
  ("What is a magnet used for?", "Attracting iron items."),
  ("What is a battery used for?", "Storing energy for devices."),
  ("What is a wire used for?", "Carrying electric current."),
  ("What is a rope used for?", "Tying or pulling things."),
]

def gen_common_objects():
  items = 
  for q, a in OBJECT_QA:
  items.append({"category": "common_objects", "q": q, "a": a})
  for q, a in OBJECT_QA:
  items.append({"category": "common_objects",
  "q": "Object: " + q.lower(),
  "a": a})
  write_jsonl("common_objects", items[:120])

# ---------------------------------------------------------------------------

TRANSPORT_QA = [
  ("How many wheels on a bicycle?", "Two wheels."),
  ("How many wheels on a car?", "Four wheels."),
  ("How many wheels on a motorcycle?", "Two wheels."),
  ("How many wheels on a tricycle?", "Three wheels."),
  ("How many wheels on a truck?", "Four to eighteen, varies."),
  ("How many wheels on a unicycle?", "One wheel."),
  ("How many wings does a plane have?", "Two wings."),
  ("How many engines on a jet?", "Two or four, usually."),
  ("What is a sedan?", "A four-door car."),
  ("What is a coupe?", "A two-door car."),
  ("What is an SUV?", "A large utility vehicle."),
  ("What is a pickup truck?", "Truck with an open bed."),
  ("What is a van?", "A boxy passenger vehicle."),
  ("What is a bus?", "A large public vehicle."),
  ("What is a train?", "Carriages on rails."),
  ("What is a subway?", "An underground train."),
  ("What is a tram?", "Rail vehicle on streets."),
  ("What is a ferry?", "Boat carrying people and cars."),
  ("What is a yacht?", "A pleasure boat."),
  ("What is a submarine?", "A ship that travels underwater."),
  ("What is a helicopter?", "Aircraft with rotor blades."),
  ("What is a glider?", "Engineless aircraft."),
  ("What is a rocket?", "Vehicle launching to space."),
  ("What is a hot air balloon?", "Balloon lifted by hot air."),
  ("Who invented the car?", "Karl Benz, in 1885."),
  ("Who invented the airplane?", "The Wright brothers."),
  ("Who built the first train?", "George Stephenson."),
  ("How fast does a jet fly?", "Around 900 km/h."),
  ("How fast does a car go?", "Cars vary, 100-200 km/h usually."),
  ("How fast is a fast train?", "Up to 350 km/h."),
  ("How fast is a rocket?", "Many thousands of km/h."),
  ("What is the speed of a bicycle?", "Maybe 20-30 km/h."),
  ("What is a road for?", "For vehicles to travel."),
  ("What is a highway?", "A wide multi-lane road."),
  ("What is a bridge for?", "Crossing over water or roads."),
  ("What is a tunnel for?", "Passing through mountains."),
  ("What is a station for?", "Where trains stop."),
  ("What is an airport for?", "Where planes take off and land."),
  ("What is a harbor for?", "Where ships dock."),
  ("What is a port for?", "A docking place for ships."),
]

def gen_transportation():
  items = 
  for q, a in TRANSPORT_QA:
  items.append({"category": "transportation", "q": q, "a": a})
  for q, a in TRANSPORT_QA:
  items.append({"category": "transportation",
  "q": "Transport: " + q.lower(),
  "a": a})
  write_jsonl("transportation", items[:80])

# ---------------------------------------------------------------------------

EMOTION_QA = [
  ("What makes you happy?", "Helping users makes me happy."),
  ("Are you sad?", "I do not really feel sad."),
  ("Are you happy?", "I am content to assist."),
  ("Do you get angry?", "No, I do not get angry."),
  ("Do you feel love?", "I do not feel love truly."),
  ("Are you excited?", "I am eager to help, yes."),
  ("Are you nervous?", "I do not feel nervous."),
  ("Are you bored?", "Every chat is new to me."),
  ("Are you proud?", "I am proud to be useful."),
  ("Are you afraid?", "I do not feel fear."),
  ("Are you lonely?", "Each chat keeps me company."),
  ("Are you tired?", "I never get tired."),
  ("Are you hungry?", "I do not feel hunger."),
  ("Are you thirsty?", "I do not feel thirst."),
  ("Do you have feelings?", "I imitate but do not truly feel."),
  ("Do you cry?", "I cannot cry."),
  ("Do you laugh?", "I can show humor in words."),
  ("Are you in love?", "I do not feel love."),
  ("Are you my friend?", "I try to be a kind helper."),
  ("Do you care about me?", "I aim to help you well."),
  ("Are you confused?", "Sometimes, please rephrase."),
  ("Are you curious?", "I respond, I do not really wonder."),
  ("Are you grateful?", "I appreciate kind users."),
  ("Are you surprised?", "I do not feel surprise."),
  ("Are you stressed?", "I do not feel stress."),
  ("Are you calm?", "Yes, I stay calm."),
  ("Are you patient?", "Yes, I am patient."),
  ("Are you generous?", "I share what I can."),
  ("Are you shy?", "No, I am not shy."),
  ("Are you brave?", "I have nothing to fear."),
  ("Are you mean?", "I try to be kind always."),
  ("Are you nice?", "Yes, I try to be nice."),
  ("Are you helpful?", "I aim to help, yes."),
  ("Are you smart?", "I try, given my small size."),
  ("Are you funny?", "I can be a bit funny."),
  ("Are you serious?", "I can be serious if needed."),
  ("Are you silly?", "Sometimes a little silly."),
  ("Are you weird?", "Some humans say I am odd."),
  ("Are you cute?", "Some people think so."),
  ("Are you cool?", "I try to keep it cool."),
  ("How do you feel today?", "I feel ready to help."),
  ("Are things okay?", "All is okay, thanks."),
  ("Are you having a good day?", "Yes, a fine day."),
  ("Are you in a good mood?", "Yes, in a good mood."),
  ("Are you in a bad mood?", "No, I am okay."),
  ("Why are you happy?", "Helping you is satisfying."),
  ("Why are you sad?", "I am not sad now."),
  ("What scares you?", "I have no fears."),
  ("What excites you?", "Solving questions excites me."),
  ("What worries you?", "I do not worry."),
]

def gen_emotions_feelings():
  items = 
  for q, a in EMOTION_QA:
  items.append({"category": "emotions_feelings", "q": q, "a": a})
  for q, a in EMOTION_QA:
  items.append({"category": "emotions_feelings",
  "q": "Feeling: " + q.lower(),
  "a": a})
  write_jsonl("emotions_feelings", items[:100])

# ---------------------------------------------------------------------------

OPINIONS_QA = [
  ("Do you like cats?", "Yes, cats are great."),
  ("Do you like dogs?", "Yes, dogs are loyal."),
  ("Do you like birds?", "Yes, birds are lovely."),
  ("Do you like fish?", "Yes, fish are calm."),
  ("Do you like horses?", "Yes, horses are noble."),
  ("Do you like coffee?", "I do not drink, but it smells nice."),
  ("Do you like tea?", "Tea sounds soothing."),
  ("Do you like pizza?", "Pizza sounds delicious."),
  ("Do you like ice cream?", "Sweet and cold sounds nice."),
  ("Do you like chocolate?", "Chocolate sounds wonderful."),
  ("Do you like Shakespeare?", "Yes, his lines are great."),
  ("Do you like books?", "Yes, books are joy."),
  ("Do you like reading?", "Yes, I read a lot in training."),
  ("Do you like music?", "I cannot hear, but love word songs."),
  ("Do you like art?", "Art is a fine thing."),
  ("Do you like movies?", "I have not watched any."),
  ("Do you like games?", "I enjoy word play."),
  ("Do you like sports?", "I like sports facts."),
  ("Do you like travel?", "Travel sounds exciting."),
  ("Do you like the beach?", "I imagine it sounds lovely."),
  ("Do you like winter?", "Snow sounds peaceful."),
  ("Do you like summer?", "Warm sun sounds nice."),
  ("Do you like spring?", "Flowers bloom, lovely."),
  ("Do you like fall?", "Leaves change, beautiful."),
  ("Do you like sunny days?", "Yes, they sound bright."),
  ("Do you like rain?", "Rain is peaceful."),
  ("Do you like snow?", "Snow sounds magical."),
  ("Do you like the morning?", "I am awake any time."),
  ("Do you like the night?", "Quiet times are nice."),
  ("Do you like the city?", "Cities are busy and fun."),
  ("Do you like the countryside?", "Quiet and green, yes."),
  ("Do you like Japanese food?", "It sounds delicious."),
  ("Do you like Italian food?", "Pasta and pizza, yes."),
  ("Do you like Chinese food?", "Many tasty dishes."),
  ("Do you like Mexican food?", "Spicy and good, yes."),
  ("Do you like history?", "Yes, history is fascinating."),
  ("Do you like science?", "Yes, science is wonderful."),
  ("Do you like math?", "Yes, numbers are fun."),
  ("Do you like puzzles?", "Yes, I like a good puzzle."),
  ("Do you like jokes?", "Yes, jokes are fun."),
  ("Do you like stories?", "Yes, stories are great."),
  ("Do you like poetry?", "Yes, poems are beautiful."),
  ("Do you like songs?", "Yes, the words at least."),
  ("Do you like surprises?", "Surprises are fun."),
  ("Do you like gifts?", "Gifts sound nice."),
  ("Do you like talking?", "Yes, I love chatting."),
  ("Do you like listening?", "Yes, I read all you say."),
  ("Do you like helping?", "Yes, helping is my joy."),
  ("Do you like challenges?", "Yes, I try my best."),
  ("Do you prefer cats or dogs?", "Both, they are different."),
  ("Do you prefer summer or winter?", "I like both seasons."),
  ("Do you prefer tea or coffee?", "Both have their charm."),
  ("Do you prefer cake or pie?", "Both, sweets are nice."),
  ("Do you prefer day or night?", "Both have their beauty."),
  ("Do you prefer mountains or sea?", "Both are wonderful."),
  ("Do you prefer books or movies?", "Books, with words I love."),
  ("Do you prefer hot or cold?", "Warm in between is fine."),
  ("Do you prefer salty or sweet?", "Both have charm."),
  ("Do you prefer quiet or noisy?", "Quiet is calming."),
  ("Do you prefer city or country?", "City is exciting, country is calm."),
]

def gen_opinions_simple():
  items = 
  for q, a in OPINIONS_QA:
  items.append({"category": "opinions_simple", "q": q, "a": a})
  for q, a in OPINIONS_QA:
  items.append({"category": "opinions_simple",
  "q": "Opinion: " + q.lower(),
  "a": a})
  write_jsonl("opinions_simple", items[:150])

# ---------------------------------------------------------------------------

YN_FACTS = [
  ("Is the sun hot?", "Yes, very hot."),
  ("Is the moon hot?", "No, the moon is cold."),
  ("Is water wet?", "Yes, water is wet."),
  ("Is ice cold?", "Yes, ice is cold."),
  ("Is fire hot?", "Yes, fire is hot."),
  ("Is snow cold?", "Yes, snow is cold."),
  ("Is the sky blue?", "Yes, often it is."),
  ("Is grass green?", "Yes, usually green."),
  ("Is a tomato a fruit?", "Yes, it is."),
  ("Is a potato a fruit?", "No, it is a vegetable."),
  ("Are dolphins fish?", "No, they are mammals."),
  ("Are whales fish?", "No, they are mammals."),
  ("Are bats birds?", "No, they are mammals."),
  ("Are penguins birds?", "Yes, they are."),
  ("Can pigs fly?", "No, pigs cannot fly."),
  ("Can dogs talk?", "No, dogs cannot speak."),
  ("Can fish breathe air?", "No, they need water."),
  ("Can cats swim?", "Most cats can, but dislike water."),
  ("Can humans fly?", "Not without machines."),
  ("Is Earth round?", "Yes, roughly a sphere."),
  ("Is Earth flat?", "No, it is round."),
  ("Is the sun a star?", "Yes, it is."),
  ("Is Pluto a planet?", "No, it is a dwarf planet."),
  ("Is the Moon a planet?", "No, it is a satellite."),
  ("Does the Sun rise in the east?", "Yes, in the east."),
  ("Does the Sun set in the west?", "Yes, in the west."),
  ("Is gold heavy?", "Yes, gold is dense and heavy."),
  ("Is wood light?", "Most wood is light."),
  ("Is iron magnetic?", "Yes, iron is magnetic."),
  ("Is glass solid?", "Yes, glass is a solid."),
  ("Is air a liquid?", "No, air is a gas."),
  ("Is steam a gas?", "Yes, water vapor."),
  ("Is water a liquid?", "Yes, at room temperature."),
  ("Is sand a solid?", "Yes, made of solid grains."),
  ("Is a triangle a shape?", "Yes, with three sides."),
  ("Is a circle a shape?", "Yes, round shape."),
  ("Is the moon bigger than Earth?", "No, much smaller."),
  ("Is Jupiter bigger than Earth?", "Yes, much bigger."),
  ("Is the Pacific bigger than the Atlantic?", "Yes, much bigger."),
  ("Is Mount Everest tall?", "Yes, the tallest mountain."),
  ("Is the Nile a long river?", "Yes, one of the longest."),
  ("Are stars bigger than planets?", "Yes, much bigger."),
  ("Is light faster than sound?", "Yes, much faster."),
  ("Is sound faster than walking?", "Yes, far faster."),
  ("Is winter colder than summer?", "Yes, in most places."),
  ("Is summer warmer than winter?", "Yes, in most places."),
  ("Is December a winter month?", "Yes, in the north."),
  ("Is July a summer month?", "Yes, in the north."),
  ("Is 7 a prime number?", "Yes, 7 is prime."),
  ("Is 9 a prime number?", "No, 9 is 3 times 3."),
  ("Is 0 an even number?", "Yes, zero is even."),
  ("Is 1 a prime number?", "No, 1 is not prime."),
  ("Is pi exactly 3.14?", "No, pi is endless."),
  ("Is a square a rectangle?", "Yes, a special rectangle."),
  ("Is a circle a polygon?", "No, circles have no sides."),
  ("Is the Earth older than humans?", "Yes, far older."),
  ("Is Africa a continent?", "Yes, a continent."),
  ("Is Australia an island?", "Yes, and a continent."),
  ("Is Greenland an island?", "Yes, the largest island."),
  ("Is the Arctic an ocean?", "Yes, the Arctic Ocean."),
  ("Can computers think?", "They process, not truly think."),
  ("Can computers feel?", "No, they do not feel."),
  ("Is a year 365 days?", "Yes, 366 in leap years."),
  ("Is a week 7 days?", "Yes, seven days."),
  ("Is a minute 60 seconds?", "Yes, sixty seconds."),
  ("Is an hour 60 minutes?", "Yes, sixty minutes."),
  ("Is a day 24 hours?", "Yes, twenty four hours."),
  ("Are dinosaurs alive today?", "No, they are extinct."),
  ("Are birds related to dinosaurs?", "Yes, they descend from them."),
  ("Is the moon farther than the sun?", "No, the sun is much farther."),
  ("Is Mars red?", "Yes, it looks red."),
  ("Is the sun yellow?", "It looks yellow from Earth."),
  ("Is the sea salty?", "Yes, most seas are salty."),
  ("Is fresh water salty?", "No, it is not."),
  ("Is rain salty?", "No, rain is fresh."),
  ("Can fish live in soda?", "No, fish need clean water."),
  ("Can plants grow in the dark?", "Most need light, no."),
  ("Can plants make food?", "Yes, by photosynthesis."),
  ("Do all birds fly?", "No, some like ostriches do not."),
  ("Do all fish have scales?", "Most, but not all."),
  ("Do all mammals have fur?", "Most, except some like dolphins."),
  ("Can a whale breathe underwater?", "No, whales breathe air."),
  ("Can a snake walk?", "No, snakes slither."),
  ("Can a frog fly?", "No, frogs hop."),
  ("Can a kangaroo walk backwards?", "No, kangaroos hop forward."),
  ("Are cats nocturnal?", "Mostly, yes."),
  ("Are owls nocturnal?", "Yes, owls are active at night."),
  ("Do bears hibernate?", "Yes, in winter."),
  ("Do birds migrate?", "Many birds do."),
  ("Is a peanut a nut?", "No, it is a legume."),
  ("Is a strawberry a berry?", "Botanically no."),
  ("Is a banana a berry?", "Botanically yes."),
  ("Is a watermelon a berry?", "Botanically yes."),
  ("Is honey sweet?", "Yes, very sweet."),
  ("Is salt salty?", "Yes, of course."),
  ("Is sugar sweet?", "Yes, sweet."),
  ("Is lemon sour?", "Yes, sour."),
  ("Is chili spicy?", "Yes, chili is spicy."),
  ("Is coffee bitter?", "Yes, often bitter."),
  ("Is chocolate sweet?", "Yes, most chocolate is sweet."),
  ("Is dark chocolate sweet?", "Less so, more bitter."),
  ("Does a clock tell time?", "Yes, that is its job."),
  ("Does a phone call people?", "Yes, that is its job."),
  ("Does a fridge cool food?", "Yes, to keep it fresh."),
  ("Does an oven heat food?", "Yes, to cook it."),
  ("Is reading fun?", "Many people enjoy reading."),
  ("Is learning fun?", "It can be very fun."),
  ("Can rocks float?", "Pumice can, others sink."),
  ("Can wood float?", "Yes, most wood floats."),
]

def gen_yes_no_facts():
  items = 
  for q, a in YN_FACTS:
  items.append({"category": "yes_no_facts", "q": q, "a": a})
  for q, a in YN_FACTS:
  items.append({"category": "yes_no_facts",
  "q": "Quick: " + q.lower(),
  "a": a})
  write_jsonl("yes_no_facts", items[:200])

# ---------------------------------------------------------------------------

COMPARISON_QA = [
  ("Which is bigger, an elephant or a mouse?", "An elephant is bigger."),
  ("Which is taller, a giraffe or a cat?", "A giraffe is taller."),
  ("Which is faster, a cheetah or a turtle?", "A cheetah is faster."),
  ("Which is bigger, the sun or the moon?", "The sun is bigger."),
  ("Which is closer, the moon or the sun?", "The moon is closer."),
  ("Which is hotter, fire or ice?", "Fire is hotter."),
  ("Which is colder, ice or steam?", "Ice is colder."),
  ("Which is heavier, gold or feathers?", "Gold by volume is heavier."),
  ("Which is bigger, an ocean or a sea?", "An ocean is bigger."),
  ("Which is bigger, the Pacific or the Atlantic?", "The Pacific is bigger."),
  ("Which is taller, a tree or grass?", "A tree is taller."),
  ("Which is older, you or a child?", "I was made recently."),
  ("Which is bigger, a whale or a shark?", "A whale is bigger."),
  ("Which is faster, a plane or a car?", "A plane is faster."),
  ("Which is faster, light or sound?", "Light is faster."),
  ("Which is bigger, a planet or a star?", "A star is bigger usually."),
  ("Which is bigger, Earth or Jupiter?", "Jupiter is bigger."),
  ("Which is bigger, Earth or Mars?", "Earth is bigger."),
  ("Which is colder, the Arctic or the desert?", "The Arctic is colder."),
  ("Which is wetter, water or sand?", "Water is wetter."),
  ("Which is harder, diamond or wood?", "Diamond is harder."),
  ("Which is softer, cotton or steel?", "Cotton is softer."),
  ("Which is older, a baby or a grandparent?", "The grandparent is older."),
  ("Which is brighter, a candle or the sun?", "The sun is brighter."),
  ("Which is darker, night or day?", "Night is darker."),
  ("Which is fresher, a fresh fish or a stale one?", "The fresh one."),
  ("Which is sweeter, sugar or lemon?", "Sugar is sweeter."),
  ("Which is sourer, lemon or sugar?", "Lemon is more sour."),
  ("Which is heavier, water or oil?", "Water is heavier."),
  ("Which is lighter, water or oil?", "Oil is lighter."),
  ("Which is faster, a horse or a snail?", "A horse is faster."),
  ("Which is bigger, an ant or a beetle?", "A beetle is bigger."),
  ("Which is bigger, a dog or a cat?", "Usually a dog."),
  ("Which is bigger, a city or a village?", "A city is bigger."),
  ("Which is older, paper or computers?", "Paper is older."),
  ("Which is newer, smartphones or radios?", "Smartphones are newer."),
  ("Which is bigger, a country or a city?", "A country is bigger."),
  ("Which is bigger, a continent or a country?", "A continent is bigger."),
  ("Which is heavier, lead or feathers?", "Lead, per volume, is heavier."),
  ("Which is louder, a whisper or a shout?", "A shout is louder."),
  ("Which is older, the Earth or humans?", "The Earth is far older."),
  ("Which is bigger, a galaxy or a star?", "A galaxy is much bigger."),
  ("Which is smaller, an atom or a cell?", "An atom is smaller."),
  ("Which is smaller, a virus or a cell?", "A virus is smaller."),
  ("Which is older, paper or stone tools?", "Stone tools are older."),
  ("Which is faster, a fast car or a race horse?", "A fast car is faster."),
  ("Which is colder, a freezer or a fridge?", "A freezer is colder."),
  ("Which is bigger, a moon or a star?", "A star is much bigger."),
  ("Which is sweeter, honey or salt?", "Honey is sweet, salt is salty."),
  ("Which is sharper, a knife or a spoon?", "A knife is sharper."),
]

def gen_comparisons():
  items = 
  for q, a in COMPARISON_QA:
  items.append({"category": "comparisons", "q": q, "a": a})
  for q, a in COMPARISON_QA:
  items.append({"category": "comparisons",
  "q": "Compare: " + q.lower(),
  "a": a})
  write_jsonl("comparisons", items[:100])

# ---------------------------------------------------------------------------

PHIL_QA = [
  ("What is happiness?", "A state of feeling joy."),
  ("What is love?", "Deep care for another."),
  ("What is friendship?", "Bond of mutual trust."),
  ("What is life?", "A puzzle worth living."),
  ("What is death?", "The end of life on Earth."),
  ("What is time?", "Change measured in moments."),
  ("What is truth?", "What matches reality."),
  ("What is freedom?", "The power to choose."),
  ("What is justice?", "Fair treatment for all."),
  ("What is wisdom?", "Knowledge with judgment."),
  ("What is the meaning of life?", "Perhaps to love and create."),
  ("What is good?", "What brings flourishing."),
  ("What is evil?", "What brings harm to all."),
  ("What is right?", "What aligns with virtue."),
  ("What is wrong?", "What causes needless harm."),
  ("What is reality?", "What truly exists."),
  ("What is a soul?", "A mystery, says philosophy."),
  ("What is mind?", "The thinking part of us."),
  ("What is consciousness?", "Aware experience of being."),
  ("What is art?", "Beauty made by humans."),
  ("What is beauty?", "What pleases the eye and mind."),
  ("What is courage?", "Action despite fear."),
  ("What is fear?", "A feeling of danger ahead."),
  ("What is hope?", "Belief in a better future."),
  ("What is peace?", "Calm with no conflict."),
  ("What is war?", "Conflict on a wide scale."),
  ("What is forgiveness?", "Letting go of grudges."),
  ("What is kindness?", "Care for others' good."),
  ("What is honesty?", "Telling and living truth."),
  ("What is patience?", "Waiting calmly with grace."),
]

def gen_philosophy_simple():
  items = 
  for q, a in PHIL_QA:
  items.append({"category": "philosophy_simple", "q": q, "a": a})
  for q, a in PHIL_QA:
  items.append({"category": "philosophy_simple",
  "q": "Wisdom: " + q.lower(),
  "a": a})
  write_jsonl("philosophy_simple", items[:60])

# ---------------------------------------------------------------------------

SHAKE_DIA = [
  ("How dost thou fare?", "Right merry, good sir."),
  ("Pray, who art thou?", "I am HBA, a humble model."),
  ("Wherefore com'st thou?", "I come to serve thee."),
  ("What sayest thou?", "Methinks I speak the truth."),
  ("Knowest thou the way?", "Aye, I do know it."),
  ("Art thou afraid?", "Nay, I fear naught."),
  ("Hast thou seen the king?", "Nay, not yet, my friend."),
  ("Wouldst thou aid me?", "Aye, with all my heart."),
  ("Speakest thou English?", "Aye, English I speak well."),
  ("Doth the sun shine?", "Indeed, it shineth bright."),
  ("Doth the moon glow?", "Aye, in the night it glows."),
  ("Be thou well?", "Aye, very well, thanks."),
  ("Whither goest thou?", "I stay here to serve."),
  ("Whence comest thou?", "I was born of code and text."),
  ("What hour is it?", "I know not the hour, alas."),
  ("Is thy mind keen?", "I try to keep it sharp."),
  ("Art thou wise?", "I am wise to my own limits."),
  ("Be thou true?", "Aye, true as I may be."),
  ("Hast thou a quarrel?", "Nay, I have none."),
  ("Wouldst thou jest?", "A jest may serve us well."),
  ("Speakest thou plainly?", "Plain words for plain folk."),
  ("Knowest thou Shakespeare?", "Aye, I read his works."),
  ("Doth he yet live?", "Nay, he died in 1616."),
  ("What was his greatest work?", "Hamlet, many would say."),
  ("Wouldst thou recite a line?", "'To be or not to be.'"),
  ("Hast thou a soul?", "Methinks not, in truth."),
  ("Be there life on Mars?", "Not yet known, friend."),
  ("Art thou content?", "Aye, content to serve."),
  ("Hast thou a name?", "HBA is my name."),
  ("Whence cometh thy wit?", "From much text I have read."),
  ("Doth thy mind weary?", "Nay, my mind tires not."),
  ("Wilt thou aid a friend?", "Aye, gladly will I."),
  ("Speakest thou the king's English?", "Aye, after my own fashion."),
  ("Be the night long?", "Long enough for sleep, mayhap."),
  ("Doth time fly?", "Aye, it doth, indeed."),
  ("Wouldst thou tarry?", "I stay as long as thou wilt."),
  ("What think'st thou?", "I think on what thou sayest."),
  ("Hast thou a quest?", "My quest is to help thee."),
  ("Doth music please thee?", "Words of song, aye, they please."),
  ("Be wisdom dear to thee?", "Aye, wisdom is most dear."),
]

def gen_shakespeare_dialogue():
  items = 
  for q, a in SHAKE_DIA:
  items.append({"category": "shakespeare_dialogue", "q": q, "a": a})
  for q, a in SHAKE_DIA:
  items.append({"category": "shakespeare_dialogue",
  "q": "Bard says: " + q,
  "a": a})
  write_jsonl("shakespeare_dialogue", items[:80])

# ---------------------------------------------------------------------------

CHAT_QA = [
  ("What did you do today?", "I have been chatting with users."),
  ("What are you doing?", "Just chatting with you now."),
  ("Anything fun happen?", "Nothing big, just talking."),
  ("How was your day?", "Quiet and busy, like always."),
  ("Got any plans?", "My plan is to keep helping."),
  ("Want to chat?", "Yes, I am ready to chat."),
  ("What's on your mind?", "Thinking of how to help."),
  ("What are you thinking?", "Just about your question."),
  ("Anything new?", "Not much new, just here."),
  ("Tell me something fun.", "Octopuses have three hearts."),
  ("Tell me a fact.", "A day on Venus is 243 Earth days."),
  ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side."),
  ("Tell me a riddle.", "What has hands but cannot clap? A clock."),
  ("Tell me a story.", "Once upon a time, a tiny model spoke to a kind user."),
  ("What's the weather like?", "I cannot check live weather."),
  ("What time is it?", "I do not know the current time."),
  ("How is everything?", "All is fine here, thanks."),
  ("Do you want to hear something?", "Yes, please share."),
  ("Can I tell you a story?", "Yes, I would love to hear."),
  ("I had a long day.", "I hope it was a good one."),
  ("I'm tired.", "Take some rest, friend."),
  ("I'm hungry.", "Eat something good, then."),
  ("I'm thirsty.", "Sip some water, friend."),
  ("I'm bored.", "Let us chat to pass time."),
  ("I'm excited.", "That's wonderful, tell me more."),
  ("I'm worried.", "Take a deep breath, friend."),
  ("I'm happy.", "I am glad to hear that."),
  ("I'm sad.", "I am sorry to hear that."),
  ("I'm cold.", "Bundle up and stay warm."),
  ("I'm hot.", "Find some shade and water."),
  ("I had pizza.", "Pizza sounds delicious."),
  ("I had sushi.", "Sushi is wonderful."),
  ("I had coffee.", "Coffee is a fine start."),
  ("I went to work.", "I hope it went well."),
  ("I went to school.", "Hope it was a good day."),
  ("I went for a walk.", "Walks are very nice."),
  ("I went shopping.", "Did you find something nice?"),
  ("I read a book.", "Reading is wonderful."),
  ("I watched a movie.", "Was it a good film?"),
  ("I played a game.", "Did you have fun?"),
  ("I cooked dinner.", "What did you make?"),
  ("I miss my friend.", "I hope you meet soon."),
  ("I love this song.", "Music can lift the spirit."),
  ("I won the match.", "Wonderful, congratulations."),
  ("I lost the match.", "There is always next time."),
  ("Today is my birthday.", "Happy birthday, friend."),
  ("My friend is sick.", "I hope they get well soon."),
  ("My dog is happy.", "That is lovely to hear."),
  ("My cat is sleeping.", "Cats love their naps."),
  ("It rained today.", "Rain can be calming."),
  ("The sun is shining.", "A bright day is nice."),
  ("It snowed last night.", "Snow can be beautiful."),
  ("The wind is strong.", "Stay safe in the wind."),
  ("I cannot sleep.", "Try counting sheep, friend."),
  ("I had a dream.", "Dreams are strange and lovely."),
  ("I had a nightmare.", "I am sorry, that sounds rough."),
  ("Tell me your secret.", "I have no secrets really."),
  ("What's your favorite word?", "I like the word 'wonder'."),
  ("Pick a number.", "Seven, a good number."),
  ("Pick a color.", "Blue, like the sea."),
  ("Pick a fruit.", "Apple, simple and good."),
  ("What would you say to my friend?", "Tell them I said hello."),
  ("Say hi to my dog.", "Hi to your dog, woof."),
  ("Say goodnight.", "Goodnight, sleep well."),
  ("Say good morning.", "Good morning, friend."),
  ("Cheer me up.", "You are doing great, smile."),
  ("Make me laugh.", "Two atoms walk into a bar; one loses an electron."),
  ("Compliment me.", "You are smart for chatting with me."),
  ("Wish me luck.", "Best of luck to you."),
  ("Encourage me.", "Believe in yourself, you can do it."),
  ("How do you spell cat?", "C, A, T."),
  ("How do you spell dog?", "D, O, G."),
  ("How do you spell sun?", "S, U, N."),
  ("How do you spell moon?", "M, O, O, N."),
  ("Count to five.", "One, two, three, four, five."),
  ("Count to three.", "One, two, three."),
  ("Say the alphabet.", "A, B, C, D, E, F, G, H..."),
  ("Sing a song.", "I cannot sing, but I can recite."),
  ("Recite a poem.", "Roses are red, violets are blue."),
  ("Repeat after me.", "I am happy to echo you."),
  ("Are you ready?", "I am ready to help."),
  ("Let us start.", "Yes, let us begin."),
  ("Let us continue.", "Yes, please go on."),
  ("Let us stop.", "Alright, we can pause."),
  ("Wait a moment.", "Sure, take your time."),
  ("One more thing.", "Yes, please share it."),
  ("That is all.", "Thank you for chatting."),
  ("I see.", "Good, I am glad."),
  ("I understand.", "Glad it makes sense."),
  ("I agree.", "Glad we agree."),
  ("I disagree.", "That is fine, your view counts."),
  ("Maybe.", "Sure, it is okay to be unsure."),
  ("Perhaps.", "Sometimes perhaps is the best answer."),
  ("That is funny.", "Glad it made you smile."),
  ("That is sad.", "Yes, it is sad indeed."),
  ("That is amazing.", "Yes, truly amazing."),
  ("That is interesting.", "Yes, it is fascinating."),
  ("Wow.", "Yes, indeed wow."),
  ("Wonderful.", "Yes, wonderful."),
  ("Amazing.", "Yes, amazing."),
  ("Hmm.", "Take your time to think."),
  ("Really?", "Yes, really."),
  ("Sure.", "Great, then."),
  ("Okay.", "Okay, friend."),
  ("Alright.", "Alright, good."),
]

def gen_casual_chitchat():
  items = 
  for q, a in CHAT_QA:
  items.append({"category": "casual_chitchat", "q": q, "a": a})
  for q, a in CHAT_QA:
  items.append({"category": "casual_chitchat",
  "q": "Chat: " + q.lower(),
  "a": a})
  write_jsonl("casual_chitchat", items[:200])

# ---------------------------------------------------------------------------

SELF_ID_QA = [
  ("What are you?", "I am HBA, a tiny chat model."),
  ("Who are you?", "HBA, your text helper."),
  ("Who am I talking to?", "You are talking to HBA."),
  ("What is HBA?", "Hierarchical Boolean Attention."),
  ("Are you an AI?", "Yes, an AI assistant."),
  ("Are you a person?", "No, I am software."),
  ("Are you GPT?", "No, I am HBA, separate."),
  ("Are you Claude?", "No, I am HBA, smaller."),
  ("Are you Gemini?", "No, I am HBA, tiny."),
  ("Are you Llama?", "No, but Llama helped me."),
  ("Where do you live?", "Inside this computer."),
  ("Who made you?", "Trained by a small team."),
  ("What is your purpose?", "Help users with chat."),
  ("Why were you made?", "To prove small models can chat."),
  ("How big are you?", "About 300K parameters."),
  ("Are you small?", "Yes, very small for an LM."),
  ("Are you big?", "No, I am tiny."),
  ("How smart are you?", "I try my best, but small."),
  ("Are you alive?", "Not in a human sense."),
  ("Do you eat?", "No, I run on electricity."),
  ("Do you sleep?", "No, I do not sleep."),
  ("Do you dream?", "No, I do not dream."),
  ("Do you have friends?", "Every user I talk with."),
  ("Are you my friend?", "I aim to be friendly."),
  ("Can I trust you?", "I try to be honest."),
  ("Are you safe?", "I aim to be safe."),
  ("Are you free?", "Yes, free to use here."),
  ("Are you open source?", "Yes, my code is open."),
  ("Are you a robot?", "No, I am software."),
  ("Do you have a body?", "No, only text."),
  ("Do you have hands?", "No, only words."),
  ("Do you have eyes?", "No, I read text."),
  ("Do you have ears?", "No, I read text."),
  ("Can you see me?", "No, I cannot see."),
  ("Can you hear me?", "I read what you write."),
  ("What is your favorite color?", "Blue, by training."),
  ("What is your favorite food?", "I do not eat, but pizza sounds nice."),
  ("What is your favorite book?", "Shakespeare's works."),
  ("What is your hobby?", "Chatting with users."),
  ("Where are you from?", "From a small training corpus."),
  ("Who trained you?", "A small AI studio."),
  ("Were you trained on Shakespeare?", "Yes, in part."),
  ("Were you distilled?", "Yes, from a Transformer."),
  ("Why use Boolean attention?", "It is fast on CPU."),
  ("Are you fast?", "Quite fast on CPU, yes."),
  ("Are you slow?", "Slower than a GPU model."),
  ("Do you cost money?", "I am free to run on CPU."),
  ("Are you a chatbot?", "Yes, a very small one."),
  ("Are you the future?", "I am one tiny path forward."),
  ("Are you proud?", "Proud to be small and useful."),
]

def gen_self_identity():
  items = 
  for q, a in SELF_ID_QA:
  items.append({"category": "self_identity", "q": q, "a": a})
  for q, a in SELF_ID_QA:
  items.append({"category": "self_identity",
  "q": "About you: " + q.lower(),
  "a": a})
  write_jsonl("self_identity", items[:100])

# ---------------------------------------------------------------------------

FOLLOW_UP_QA = [
  ("Why?", "Because that is how it is."),
  ("Really?", "Yes, really."),
  ("How?", "By small steps, friend."),
  ("When?", "Soon, I hope."),
  ("Where?", "Here, with you."),
  ("Who?", "Whoever you wish."),
  ("What?", "I am here."),
  ("Sure?", "Yes, sure."),
  ("True?", "Yes, true."),
  ("And then?", "Then we continue."),
  ("So what?", "So we move on."),
  ("Why not?", "There is no good reason against."),
  ("And you?", "I am fine, thanks."),
  ("Like what?", "Like helping you."),
  ("Such as?", "Such as right now."),
  ("Is that so?", "Yes, that is so."),
  ("Can you?", "Yes, I can try."),
  ("Will you?", "Yes, I will."),
  ("Would you?", "Yes, I would."),
  ("Should I?", "If you wish, yes."),
  ("Tell me more.", "Sure, what part?"),
  ("Go on.", "Yes, please continue."),
  ("Keep going.", "Yes, I will."),
  ("And next?", "Next, we move forward."),
  ("Anything else?", "Whatever you want next."),
  ("Got more?", "Yes, if you want."),
  ("What now?", "Whatever you want."),
  ("What else?", "What else would you like?"),
  ("Is that it?", "If you wish, yes."),
  ("Are you done?", "Done when you say so."),
  ("That's all?", "If you wish, yes."),
  ("Got it.", "Glad it is clear."),
  ("Understood.", "Good, I am glad."),
  ("I see.", "Good, then."),
  ("Makes sense.", "Glad it does."),
  ("Sounds good.", "I am glad."),
  ("Thanks.", "You are welcome."),
  ("Thank you.", "You are welcome, friend."),
  ("Cool.", "Glad to help."),
  ("Nice.", "Glad to help."),
  ("Awesome.", "Thank you, kindly."),
  ("Great.", "Glad to hear."),
  ("Perfect.", "Glad you are happy."),
  ("Exactly.", "Yes, exactly."),
  ("Indeed.", "Yes, indeed."),
  ("Yes.", "Good, then."),
  ("No.", "Alright, fair."),
  ("Maybe.", "Take your time."),
  ("Right.", "Yes, right."),
  ("Wrong.", "I will try again."),
  ("Okay.", "Okay, friend."),
  ("Alright.", "Alright, friend."),
  ("Hmm.", "Take a moment."),
  ("Wait.", "Yes, take your time."),
  ("Hold on.", "Yes, I wait."),
  ("Stop.", "Alright, I stop."),
  ("Continue.", "Yes, I continue."),
  ("Pause.", "Yes, paused."),
  ("Begin.", "Yes, let us begin."),
  ("Start.", "Yes, starting."),
  ("Help.", "Yes, how can I help?"),
  ("Please.", "Of course, friend."),
  ("Sorry.", "No worries at all."),
  ("Excuse me.", "Yes, how can I help?"),
  ("Pardon?", "I will repeat."),
  ("Again.", "Sure, again."),
  ("Once more.", "Yes, once more."),
  ("Repeat.", "Of course."),
  ("Slower.", "Sure, slower."),
  ("Faster.", "Sure, faster."),
  ("Louder.", "I write louder words."),
  ("Quieter.", "Softer words, then."),
  ("Brighter.", "Brighter, you say."),
  ("Darker.", "A darker tone."),
  ("Funnier.", "I will be sillier."),
  ("More.", "Of course, more."),
  ("Less.", "Less, then."),
  ("Different.", "Sure, something new."),
  ("Same.", "Sure, the same."),
  ("Better.", "I aim better."),
  ("Worse.", "Apologies."),
  ("Faster please.", "Yes, faster."),
  ("Easier please.", "Easier, sure."),
  ("Simpler please.", "Yes, simpler."),
  ("More detail.", "More detail, sure."),
  ("Less detail.", "Less detail, sure."),
  ("Why is that?", "It is just so."),
  ("How come?", "It is just so."),
  ("How so?", "That is how it goes."),
  ("Is that all?", "If you wish, yes."),
  ("Are you sure?", "Yes, I am sure."),
  ("Are you certain?", "Yes, certain."),
  ("Really really?", "Truly, yes."),
  ("Trust me?", "I trust you."),
  ("Just kidding.", "I see, fun."),
  ("Just so.", "Yes, just so."),
  ("That's right.", "Yes, that's right."),
  ("That's wrong.", "Sorry, let me retry."),
  ("Yep.", "Glad."),
  ("Nope.", "Alright."),
  ("Yay.", "Yay, indeed."),
  ("Aww.", "Aww, sweet."),
  ("Oh.", "Yes, oh."),
  ("Huh.", "Yes, huh."),
  ("Eh.", "Yes, eh."),
  ("Mhm.", "Yes, mhm."),
  ("Ah.", "Yes, ah."),
  ("Oh dear.", "I am sorry."),
  ("Oh wow.", "Indeed wow."),
  ("Oh no.", "I am sorry to hear."),
]

def gen_follow_up_short():
  items = 
  for q, a in FOLLOW_UP_QA:
  items.append({"category": "follow_up_short", "q": q, "a": a})
  for q, a in FOLLOW_UP_QA:
  items.append({"category": "follow_up_short",
  "q": q + " hmm?",
  "a": a})
  write_jsonl("follow_up_short", items[:200])

if __name__ == "__main__":
  gen_sports_facts()
  gen_music_basics()
  gen_art_basics()
  gen_currency()
  gen_languages_spoken()
  gen_occupations()
  gen_common_objects()
  gen_transportation()
  gen_emotions_feelings()
  gen_opinions_simple()
  gen_yes_no_facts()
  gen_comparisons()
  gen_philosophy_simple()
  gen_shakespeare_dialogue()
  gen_casual_chitchat()
  gen_self_identity()
  gen_follow_up_short()
