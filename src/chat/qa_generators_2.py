"""
Q&A 生成器 2: science_*, history_*
"""

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "qa"
OUT_DIR.mkdir(exist_ok=True)

random.seed(43)

def write_jsonl(category: str, items: list):
  path = OUT_DIR / f"{category}.jsonl"
  with path.open('w', encoding='utf-8') as f:
  for it in items:
  f.write(json.dumps(it, ensure_ascii=False) + '\n')
  print(f"[{category}] {len(items)} pairs -> {path}")

# ---------------------------------------------------------------------------
# science_basics: 物理・化学・生物の基本事実
# ---------------------------------------------------------------------------

SCIENCE_BASICS_FACTS = [
  ("What gas do plants breathe in?", "Plants breathe in carbon dioxide."),
  ("What gas do plants give off?", "Plants give off oxygen."),
  ("What gas do humans breathe in?", "Humans breathe in oxygen."),
  ("What gas do humans breathe out?", "Humans breathe out carbon dioxide."),
  ("What is photosynthesis?", "Plants making food from sunlight."),
  ("What is gravity?", "A force that pulls objects down."),
  ("Who discovered gravity?", "Isaac Newton discovered it."),
  ("What is the speed of light?", "About 300 thousand km per second."),
  ("What is sound?", "A wave of pressure in air."),
  ("What is heat?", "A form of energy from motion."),
  ("What is the boiling point of water?", "100 degrees Celsius."),
  ("What is the freezing point of water?", "0 degrees Celsius."),
  ("What is H2O?", "It is the chemical for water."),
  ("What is CO2?", "Carbon dioxide gas."),
  ("What is O2?", "Oxygen gas."),
  ("What is NaCl?", "Common table salt."),
  ("What is salt made of?", "Sodium and chloride."),
  ("What is rust?", "Iron reacting with oxygen."),
  ("What is fire made of?", "Hot glowing gas called plasma."),
  ("Why is the sky blue?", "Light scatters in the air."),
  ("What is electricity?", "Flow of electric charge."),
  ("What is a circuit?", "A loop where current flows."),
  ("What is magnetism?", "A force from moving charges."),
  ("What is energy?", "The ability to do work."),
  ("What is matter?", "Anything with mass and volume."),
  ("What is an atom?", "The smallest unit of matter."),
  ("What is a molecule?", "Two or more atoms joined."),
  ("What is a cell?", "The smallest unit of life."),
  ("What is DNA?", "Code that makes living things."),
  ("What is a gene?", "A unit of inherited DNA."),
  ("What is evolution?", "Slow change in species over time."),
  ("Who proposed evolution?", "Charles Darwin proposed it."),
  ("What is a virus?", "Tiny code that infects cells."),
  ("What is bacteria?", "Tiny living one-cell organism."),
  ("What is the brain?", "The control center of the body."),
  ("What is the heart?", "It pumps blood through the body."),
  ("What is blood?", "Red fluid that carries oxygen."),
  ("What is a vaccine?", "A shot that teaches immunity."),
  ("What is sound made of?", "Vibrating waves of air."),
  ("How fast is sound?", "About 343 meters per second."),
  ("What is a rainbow?", "Light split into colors by water."),
  ("How many colors in a rainbow?", "Seven main colors."),
  ("What are clouds made of?", "Tiny drops of water."),
  ("What is wind?", "Air moving due to pressure."),
  ("What is rain?", "Water falling from clouds."),
  ("What is snow?", "Frozen water from clouds."),
  ("What is lightning?", "Electric spark in the sky."),
  ("What is thunder?", "Sound after a lightning bolt."),
  ("What causes day and night?", "Earth spinning on its axis."),
  ("What causes seasons?", "Earth's tilt as it orbits the sun."),
  ("How long is a year?", "About 365 days."),
  ("How long is a day?", "About 24 hours."),
  ("What is a year?", "One trip of Earth around the sun."),
  ("What is the moon?", "Earth's natural satellite."),
  ("Why does the moon shine?", "It reflects sunlight."),
  ("What is the sun?", "A star at our system's center."),
  ("What is a star?", "A huge ball of burning gas."),
  ("How many stars in our galaxy?", "About 100 billion stars."),
  ("What is the Milky Way?", "Our home galaxy of stars."),
  ("What is a galaxy?", "A vast group of stars."),
  ("How many galaxies are there?", "Many billions."),
  ("What is a black hole?", "A region where gravity wins."),
  ("What is the Big Bang?", "Beginning of the universe."),
  ("How old is the universe?", "About 13.8 billion years old."),
  ("How old is the Earth?", "About 4.5 billion years old."),
  ("What is an ecosystem?", "A web of life and its place."),
  ("What is a habitat?", "Where an organism lives."),
  ("What is a food chain?", "Who eats whom in nature."),
  ("What is a predator?", "An animal that hunts others."),
  ("What is prey?", "An animal that is hunted."),
  ("What are mammals?", "Warm-blooded animals with fur."),
  ("What are reptiles?", "Cold-blooded animals with scales."),
  ("What are amphibians?", "Animals with moist skin like frogs."),
  ("What are insects?", "Animals with six legs and feelers."),
  ("What are fish?", "Animals with gills and scales."),
  ("What is a planet?", "A round body orbiting a star."),
  ("How many planets are there?", "Eight planets in our system."),
  ("What is the largest planet?", "Jupiter is the largest."),
  ("What is the smallest planet?", "Mercury is the smallest."),
  ("Which planet has rings?", "Saturn is most famous for rings."),
  ("What is gravity on the moon?", "About one sixth of Earth's."),
  ("What is the speed of sound?", "About 343 m/s in air."),
  ("Why does ice float?", "It is less dense than water."),
  ("Why does the sun set?", "Earth rotates away from it."),
  ("What is osmosis?", "Water moving through a membrane."),
  ("What is a force?", "A push or a pull on an object."),
  ("What is Newton's first law?", "Things at rest stay at rest."),
  ("What is Newton's second law?", "Force equals mass times acceleration."),
  ("What is Newton's third law?", "Every action has equal reaction."),
  ("What is friction?", "A force slowing rubbing objects."),
  ("What is acceleration?", "Change of velocity over time."),
  ("What is velocity?", "Speed with direction."),
  ("What is mass?", "How much matter in an object."),
  ("What is weight?", "Force of gravity on mass."),
  ("What is volume?", "Space an object takes up."),
  ("What is density?", "Mass per unit volume."),
  ("Why does iron sink?", "It is denser than water."),
  ("What is a magnet?", "An object that attracts iron."),
  ("Why is grass green?", "It contains chlorophyll."),
  ("Why are leaves green?", "Because of chlorophyll."),
  ("Why do leaves change color?", "Chlorophyll breaks down in fall."),
  ("What is a comet?", "An icy body orbiting the sun."),
  ("What is an asteroid?", "A rocky body orbiting the sun."),
  ("What is a meteor?", "A rock burning in our atmosphere."),
  ("What is a meteorite?", "A meteor that hits the ground."),
  ("What is the ozone layer?", "A layer that blocks UV light."),
  ("What is UV light?", "Light beyond violet, can burn skin."),
  ("What is a tsunami?", "A huge wave from a quake."),
  ("What causes earthquakes?", "Movement of Earth's plates."),
  ("What is a volcano?", "A vent where lava can erupt."),
  ("What is lava?", "Molten rock that flows out."),
  ("What is magma?", "Molten rock under the surface."),
  ("What is a fossil?", "Old remains preserved in rock."),
  ("What is dinosaur?", "Ancient reptile, long extinct."),
  ("When did dinosaurs die?", "About 66 million years ago."),
]

SCIENCE_PARAPHRASE = [
  lambda q: q,
  lambda q: q.replace("What is", "Tell me what is"),
  lambda q: "Quick: " + q.lower(),
]

def gen_science_basics():
  items = 
  for q, a in SCIENCE_BASICS_FACTS:
  items.append({"category": "science_basics", "q": q, "a": a})
  # 200件目標、110件 → 2倍に膨らます（言い回し変える）
  for q, a in SCIENCE_BASICS_FACTS:
  q2 = SCIENCE_PARAPHRASE[1](q)
  items.append({"category": "science_basics", "q": q2, "a": a})
  write_jsonl("science_basics", items[:200])

# ---------------------------------------------------------------------------
# science_animals
# ---------------------------------------------------------------------------

ANIMAL_FACTS = [
  ("How many legs does a spider have?", "Eight legs."),
  ("How many legs does an insect have?", "Six legs."),
  ("How many legs does an octopus have?", "Eight arms, no legs."),
  ("How many legs does a horse have?", "Four legs."),
  ("How many legs does a centipede have?", "Many legs, often dozens."),
  ("What does a cow eat?", "Cows mostly eat grass."),
  ("What does a lion eat?", "Lions eat meat."),
  ("What does a panda eat?", "Pandas eat bamboo."),
  ("What does a koala eat?", "Koalas eat eucalyptus leaves."),
  ("What does a bee make?", "Bees make honey."),
  ("What sound does a cow make?", "A cow says moo."),
  ("What sound does a dog make?", "A dog says woof."),
  ("What sound does a cat make?", "A cat says meow."),
  ("What sound does a duck make?", "A duck says quack."),
  ("What sound does an owl make?", "An owl says hoot."),
  ("What sound does a rooster make?", "A rooster says cock-a-doodle-doo."),
  ("What sound does a sheep make?", "A sheep says baa."),
  ("What sound does a horse make?", "A horse neighs or whinnies."),
  ("What is a baby cat called?", "A kitten."),
  ("What is a baby dog called?", "A puppy."),
  ("What is a baby cow called?", "A calf."),
  ("What is a baby horse called?", "A foal."),
  ("What is a baby sheep called?", "A lamb."),
  ("What is a baby pig called?", "A piglet."),
  ("What is a baby chicken called?", "A chick."),
  ("What is a baby goat called?", "A kid."),
  ("What is a baby duck called?", "A duckling."),
  ("What is a baby kangaroo called?", "A joey."),
  ("What is a baby bear called?", "A cub."),
  ("What is a baby lion called?", "A cub."),
  ("What is a baby deer called?", "A fawn."),
  ("What is a group of lions called?", "A pride."),
  ("What is a group of wolves called?", "A pack."),
  ("What is a group of fish called?", "A school."),
  ("What is a group of birds called?", "A flock."),
  ("What is a group of cows called?", "A herd."),
  ("What is a group of sheep called?", "A flock."),
  ("What is a group of ants called?", "A colony."),
  ("What is a group of bees called?", "A swarm."),
  ("Where do polar bears live?", "Polar bears live in the Arctic."),
  ("Where do kangaroos live?", "Kangaroos live in Australia."),
  ("Where do pandas live?", "Pandas live in China."),
  ("Where do penguins live?", "Penguins live near the South Pole."),
  ("Where do camels live?", "Camels live in deserts."),
  ("Where do elephants live?", "In Africa and Asia."),
  ("Where do tigers live?", "Mainly in Asia."),
  ("Where do lions live?", "Mainly in Africa."),
  ("Where do polar bears swim?", "In Arctic waters."),
  ("Which is the fastest land animal?", "The cheetah."),
  ("Which is the largest animal?", "The blue whale."),
  ("Which is the tallest animal?", "The giraffe."),
  ("Which is the largest land animal?", "The elephant."),
  ("Which is the largest cat?", "The tiger."),
  ("Which is the slowest mammal?", "The sloth."),
  ("Which is the longest snake?", "The reticulated python."),
  ("Which is the most venomous spider?", "The Sydney funnel-web."),
  ("Which is the smartest animal?", "Dolphins and apes are smart."),
  ("Are dolphins fish?", "No, dolphins are mammals."),
  ("Are whales fish?", "No, whales are mammals."),
  ("Are bats birds?", "No, bats are mammals."),
  ("Do fish breathe air?", "No, fish breathe water with gills."),
  ("How many hearts does an octopus have?", "Three hearts."),
  ("How many stomachs does a cow have?", "Four chambers."),
  ("How long do elephants live?", "About 60 to 70 years."),
  ("How long do parrots live?", "Some live 50 years or more."),
  ("Do snakes have legs?", "No, snakes have no legs."),
  ("Do snakes have ears?", "They sense vibrations, not airwaves."),
  ("Can fish blink?", "No, they have no eyelids."),
  ("Can pigs sweat?", "No, pigs cannot sweat."),
  ("Can dogs see colors?", "Yes, but fewer than humans."),
  ("Can owls turn their heads?", "Yes, almost all the way around."),
  ("What is a carnivore?", "An animal that eats meat."),
  ("What is a herbivore?", "An animal that eats plants."),
  ("What is an omnivore?", "An animal that eats both."),
  ("Are humans omnivores?", "Yes, we eat plants and meat."),
  ("What do frogs eat?", "Frogs eat insects."),
  ("What do giraffes eat?", "Giraffes eat leaves."),
  ("What do bears eat?", "Bears eat berries, fish, and meat."),
  ("What do butterflies eat?", "Butterflies sip nectar."),
  ("How do birds fly?", "Wings push air to make lift."),
  ("How do snakes move?", "They slither using muscles."),
  ("How do fish swim?", "With fins and a tail."),
  ("How do kangaroos move?", "They hop using strong legs."),
  ("What is a marsupial?", "A mammal that has a pouch."),
  ("Name a marsupial.", "Kangaroo or koala."),
  ("What is an amphibian?", "An animal living on land and water."),
  ("Name an amphibian.", "A frog or salamander."),
  ("What is a reptile?", "A cold-blooded scaly animal."),
  ("Name a reptile.", "A snake or a lizard."),
  ("What is a mammal?", "A warm-blooded furry animal."),
  ("Name a mammal.", "A dog, cat, or cow."),
  ("Why do birds migrate?", "To find food and warmth."),
  ("Why do bears hibernate?", "To save energy in winter."),
  ("Why do chameleons change color?", "To blend in or signal."),
  ("Why do fireflies glow?", "To attract mates."),
  ("Why do peacocks have tails?", "To attract females."),
  ("What animal sees best at night?", "The owl."),
  ("What is the heaviest bird?", "The ostrich."),
  ("Can ostriches fly?", "No, ostriches cannot fly."),
  ("Can penguins fly?", "No, but they swim well."),
]

def gen_science_animals():
  items = 
  for q, a in ANIMAL_FACTS:
  items.append({"category": "science_animals", "q": q, "a": a})
  # 100件 → 200件、paraphrase
  for q, a in ANIMAL_FACTS:
  q2 = "Animal quiz: " + q
  items.append({"category": "science_animals", "q": q2, "a": a})
  write_jsonl("science_animals", items[:200])

# ---------------------------------------------------------------------------
# science_planets
# ---------------------------------------------------------------------------

PLANET_FACTS = [
  ("How many planets in our system?", "Eight planets."),
  ("Name the planets in order.", "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune."),
  ("What is the closest planet to the sun?", "Mercury."),
  ("What is the farthest planet from the sun?", "Neptune."),
  ("Which planet is largest?", "Jupiter is largest."),
  ("Which planet is smallest?", "Mercury is smallest."),
  ("Which planet is hottest?", "Venus is hottest."),
  ("Which planet has the most moons?", "Saturn or Jupiter has many."),
  ("How many moons does Earth have?", "One moon."),
  ("How many moons does Mars have?", "Two moons."),
  ("What are Mars's moons named?", "Phobos and Deimos."),
  ("Why is Mars red?", "Iron oxide on the surface."),
  ("Is Pluto a planet?", "Now a dwarf planet."),
  ("What is Pluto now?", "A dwarf planet."),
  ("Which planet has rings?", "Saturn most famously."),
  ("Do other planets have rings?", "Yes, all gas giants do."),
  ("What is Jupiter's red spot?", "A giant storm on Jupiter."),
  ("How long is a day on Venus?", "Very long, about 243 Earth days."),
  ("How long is a year on Mercury?", "About 88 Earth days."),
  ("Is there life on Mars?", "Not confirmed yet."),
  ("Has anyone walked on Mars?", "No, not yet."),
  ("Has anyone walked on the moon?", "Yes, in 1969."),
  ("Who first walked on the moon?", "Neil Armstrong."),
  ("When did humans land on the moon?", "In July 1969."),
  ("What is the moon made of?", "Mostly rock and dust."),
  ("What is the sun made of?", "Mostly hydrogen and helium."),
  ("Is the sun a planet?", "No, the sun is a star."),
  ("How big is the sun?", "About 1.4 million km wide."),
  ("How hot is the sun?", "About 5500 C on the surface."),
  ("How far is the sun?", "About 150 million km."),
  ("How far is the moon?", "About 384 thousand km."),
  ("What is a light year?", "Distance light travels in a year."),
  ("How far is the nearest star?", "About 4.2 light years."),
  ("What is Proxima Centauri?", "The nearest star to our sun."),
  ("Are there other galaxies?", "Yes, many billions."),
  ("Is the universe expanding?", "Yes, it expands."),
  ("What is dark matter?", "Unseen mass in space."),
  ("What is dark energy?", "Force pushing space apart."),
  ("Is space cold or hot?", "Mostly very cold."),
]

def gen_science_planets():
  items = 
  for q, a in PLANET_FACTS:
  items.append({"category": "science_planets", "q": q, "a": a})
  # 40件 → 80件
  for q, a in PLANET_FACTS:
  q2 = "Space question: " + q.lower()
  items.append({"category": "science_planets", "q": q2, "a": a})
  write_jsonl("science_planets", items[:80])

# ---------------------------------------------------------------------------
# science_human_body
# ---------------------------------------------------------------------------

BODY_FACTS = [
  ("How many bones in the human body?", "About 206 bones."),
  ("How many bones do babies have?", "About 270 at birth."),
  ("How many muscles in the body?", "More than 600."),
  ("How many teeth do adults have?", "32 teeth."),
  ("How many teeth do kids have?", "20 baby teeth."),
  ("How many chambers in the heart?", "Four chambers."),
  ("How many lungs do humans have?", "Two lungs."),
  ("How many kidneys?", "Two kidneys."),
  ("How many eyes do humans have?", "Two eyes."),
  ("How many ears do humans have?", "Two ears."),
  ("How many fingers on one hand?", "Five fingers."),
  ("How many toes on one foot?", "Five toes."),
  ("How many ribs do humans have?", "Twelve pairs."),
  ("What is the largest organ?", "The skin."),
  ("What is the smallest bone?", "The stapes in the ear."),
  ("What is the strongest muscle?", "The jaw muscle."),
  ("What is the longest bone?", "The thigh bone, femur."),
  ("Where is the femur?", "In the thigh."),
  ("What pumps blood?", "The heart."),
  ("What filters blood?", "The kidneys."),
  ("What breathes air?", "The lungs."),
  ("What digests food?", "The stomach and intestines."),
  ("What controls the body?", "The brain."),
  ("What carries nerve signals?", "The spinal cord."),
  ("What is saliva?", "Liquid in the mouth for digestion."),
  ("What is bile?", "Liquid from the liver to digest fat."),
  ("What is blood?", "Red fluid carrying oxygen."),
  ("What is plasma?", "The liquid part of blood."),
  ("What are red blood cells for?", "Carrying oxygen."),
  ("What are white blood cells for?", "Fighting infection."),
  ("What does the liver do?", "Cleans toxins, stores energy."),
  ("What does the stomach do?", "Breaks down food."),
  ("What does the brain do?", "Thinks and controls."),
  ("How long is the small intestine?", "About 6 meters."),
  ("How long is the large intestine?", "About 1.5 meters."),
  ("What is the appendix?", "A small pouch in the gut."),
  ("Why do we yawn?", "Possibly to wake the brain."),
  ("Why do we sneeze?", "To clear the nose."),
  ("Why do we cough?", "To clear the throat."),
  ("Why do we hiccup?", "Spasms of the diaphragm."),
  ("What is sweat for?", "Cooling the body."),
  ("Why do we shiver?", "To warm up the body."),
  ("How many heartbeats per day?", "About 100 thousand."),
  ("How fast do nails grow?", "About 3 mm per month."),
  ("How fast does hair grow?", "About 1 cm per month."),
  ("How long do red blood cells live?", "About 120 days."),
  ("What is the largest artery?", "The aorta."),
  ("What is the body's biggest vein?", "The vena cava."),
  ("What is the cerebellum?", "Part of the brain for balance."),
  ("What is the cerebrum?", "The thinking part of the brain."),
]

def gen_science_human_body():
  items = 
  for q, a in BODY_FACTS:
  items.append({"category": "science_human_body", "q": q, "a": a})
  # 50件 → 100件
  for q, a in BODY_FACTS:
  q2 = "Human body: " + q.lower()
  items.append({"category": "science_human_body", "q": q2, "a": a})
  write_jsonl("science_human_body", items[:100])

# ---------------------------------------------------------------------------
# history_events
# ---------------------------------------------------------------------------

HIST_EVENTS = [
  ("When did World War II end?", "In 1945."),
  ("When did World War I begin?", "In 1914."),
  ("When did World War II begin?", "In 1939."),
  ("When did the Berlin Wall fall?", "In 1989."),
  ("When did the moon landing happen?", "In 1969."),
  ("When was the Declaration of Independence?", "In 1776."),
  ("When was the French Revolution?", "In 1789."),
  ("When did the Roman Empire fall?", "In 476 AD in the west."),
  ("When was Rome founded?", "In 753 BC by tradition."),
  ("When did the Cold War end?", "In 1991."),
  ("When did the Soviet Union collapse?", "In 1991."),
  ("When did Columbus reach America?", "In 1492."),
  ("When did Magna Carta sign?", "In 1215."),
  ("When did the printing press begin?", "Around 1440."),
  ("Who invented the printing press?", "Johannes Gutenberg."),
  ("When was the Renaissance?", "Roughly 1400 to 1600."),
  ("When did the Industrial Revolution start?", "In the late 1700s."),
  ("When did the French Revolution end?", "Around 1799."),
  ("Who led the French Revolution?", "Many leaders, like Robespierre."),
  ("When did Napoleon lose at Waterloo?", "In 1815."),
  ("Who fought at Waterloo?", "Napoleon vs Wellington."),
  ("When was the American Civil War?", "1861 to 1865."),
  ("Who fought the American Civil War?", "The North and the South."),
  ("Who freed the slaves?", "Abraham Lincoln signed it."),
  ("When was the Russian Revolution?", "In 1917."),
  ("When did Hitler come to power?", "In 1933."),
  ("When was the atomic bomb used?", "In 1945."),
  ("Where was the first atomic bomb dropped?", "On Hiroshima."),
  ("When was the JFK assassination?", "In 1963."),
  ("When did Mahatma Gandhi die?", "In 1948."),
  ("When was the Apollo 11 mission?", "In 1969."),
  ("Who was the first man on the moon?", "Neil Armstrong."),
  ("Who was the first woman in space?", "Valentina Tereshkova."),
  ("Who was the first person in space?", "Yuri Gagarin."),
  ("When did Gagarin orbit Earth?", "In 1961."),
  ("When did the Titanic sink?", "In 1912."),
  ("When did the Great Fire of London happen?", "In 1666."),
  ("When did the Black Death strike?", "Around 1347 to 1351."),
  ("When was the Battle of Hastings?", "In 1066."),
  ("Who won at Hastings?", "William the Conqueror."),
  ("When was the Spanish Armada?", "In 1588."),
  ("Who defeated the Spanish Armada?", "England under Elizabeth I."),
  ("When did the Mayflower sail?", "In 1620."),
  ("When was Vietnam War?", "Roughly 1955 to 1975."),
  ("When was the Korean War?", "1950 to 1953."),
  ("When did India gain independence?", "In 1947."),
  ("Who led Indian independence?", "Mahatma Gandhi."),
  ("When did South Africa end apartheid?", "In 1994."),
  ("Who ended apartheid?", "Nelson Mandela led it."),
  ("When was Mandela freed?", "In 1990."),
  ("When was the EU founded?", "In 1993."),
  ("When did the Internet start?", "In the late 1960s."),
  ("Who invented the World Wide Web?", "Tim Berners-Lee."),
  ("When was the Web invented?", "In 1989."),
  ("When did the iPhone launch?", "In 2007."),
  ("Who made the iPhone?", "Steve Jobs at Apple."),
  ("When did Pearl Harbor happen?", "In 1941."),
  ("When did D-Day happen?", "In 1944."),
  ("When was the Cold War start?", "Around 1947."),
  ("When did the Cuban missile crisis happen?", "In 1962."),
  ("When did the Wright brothers fly?", "In 1903."),
  ("Who flew the first plane?", "The Wright brothers."),
  ("When was the steam engine?", "Around 1769 by Watt."),
  ("Who invented the steam engine?", "James Watt improved it."),
  ("When did Edison invent the bulb?", "In 1879."),
  ("When did Bell invent the phone?", "In 1876."),
  ("Who invented the telephone?", "Alexander Graham Bell."),
  ("Who invented the lightbulb?", "Thomas Edison."),
  ("When was the great depression?", "Started in 1929."),
  ("When was the Tunguska event?", "In 1908."),
  ("When was the Vesuvius eruption?", "In 79 AD."),
  ("What did Vesuvius destroy?", "Pompeii."),
  ("When did Brexit happen?", "Officially in 2020."),
  ("When did COVID start?", "In late 2019."),
  ("When did the EU euro start?", "In 1999."),
]

def gen_history_events():
  items = 
  for q, a in HIST_EVENTS:
  items.append({"category": "history_events", "q": q, "a": a})
  # 75件 → 150件
  for q, a in HIST_EVENTS:
  q2 = "History: " + q
  items.append({"category": "history_events", "q": q2, "a": a})
  write_jsonl("history_events", items[:150])

# ---------------------------------------------------------------------------
# history_figures
# ---------------------------------------------------------------------------

HIST_FIGURES = [
  ("Who was Napoleon?", "A French emperor and general."),
  ("Who was Cleopatra?", "Queen of ancient Egypt."),
  ("Who was Julius Caesar?", "A Roman general and dictator."),
  ("Who was Alexander the Great?", "Greek king who conquered far."),
  ("Who was Genghis Khan?", "Mongol leader who built an empire."),
  ("Who was Christopher Columbus?", "Italian explorer of 1492."),
  ("Who was George Washington?", "First president of the USA."),
  ("Who was Abraham Lincoln?", "16th US president, freed slaves."),
  ("Who was Thomas Jefferson?", "Wrote the Declaration of Independence."),
  ("Who was Benjamin Franklin?", "US founder and inventor."),
  ("Who was Martin Luther King?", "US civil rights leader."),
  ("Who was Gandhi?", "Indian peace leader."),
  ("Who was Nelson Mandela?", "Anti-apartheid leader."),
  ("Who was Winston Churchill?", "Britain's WWII prime minister."),
  ("Who was Franklin Roosevelt?", "US president during WWII."),
  ("Who was Stalin?", "Soviet leader for decades."),
  ("Who was Mao Zedong?", "Founded the People's Republic of China."),
  ("Who was Karl Marx?", "Wrote The Communist Manifesto."),
  ("Who was Adolf Hitler?", "Dictator of Nazi Germany."),
  ("Who was Albert Einstein?", "Physicist of relativity."),
  ("Who was Isaac Newton?", "Discovered laws of motion."),
  ("Who was Galileo?", "Italian astronomer of the stars."),
  ("Who was Charles Darwin?", "Proposed evolution by selection."),
  ("Who was Marie Curie?", "Pioneer of radioactivity research."),
  ("Who was Nikola Tesla?", "Inventor of AC electric power."),
  ("Who was Thomas Edison?", "American inventor, bulb and more."),
  ("Who was Leonardo da Vinci?", "Italian Renaissance polymath."),
  ("Who was Michelangelo?", "Renaissance sculptor and painter."),
  ("Who was Vincent van Gogh?", "Dutch painter of Starry Night."),
  ("Who was Pablo Picasso?", "Spanish painter, cubism founder."),
  ("Who was Mozart?", "Austrian classical composer."),
  ("Who was Beethoven?", "German classical composer."),
  ("Who was Bach?", "German Baroque composer."),
  ("Who was William Shakespeare?", "English playwright of plays."),
  ("Who was Charles Dickens?", "English novelist of A Christmas Carol."),
  ("Who was Mark Twain?", "American novelist of Tom Sawyer."),
  ("Who was Ernest Hemingway?", "American novelist of Old Man and Sea."),
  ("Who was Jane Austen?", "English novelist of Pride and Prejudice."),
  ("Who was J.K. Rowling?", "Wrote the Harry Potter series."),
  ("Who was Steve Jobs?", "Co-founder of Apple Inc."),
  ("Who was Bill Gates?", "Co-founder of Microsoft."),
  ("Who was Elon Musk?", "Founder of Tesla and SpaceX."),
  ("Who was Alan Turing?", "Father of modern computing."),
  ("Who was Ada Lovelace?", "Wrote the first algorithm."),
  ("Who was John von Neumann?", "Mathematician of computing."),
  ("Who was Alexander the Great's teacher?", "Aristotle."),
  ("Who was Plato?", "Greek philosopher, Socrates' pupil."),
  ("Who was Aristotle?", "Greek philosopher, Plato's pupil."),
  ("Who was Socrates?", "Greek philosopher of questions."),
  ("Who was Confucius?", "Chinese sage of ethics."),
  ("Who was Buddha?", "Founder of Buddhism."),
  ("Who was Muhammad?", "Founder of Islam."),
  ("Who was Jesus?", "Founder of Christianity."),
  ("Who was Moses?", "A prophet of the Old Testament."),
  ("Who was Joan of Arc?", "French saint and warrior."),
  ("Who was Queen Elizabeth I?", "Tudor queen of England."),
  ("Who was Queen Elizabeth II?", "Long reigning UK queen."),
  ("Who was Queen Victoria?", "19th century UK queen."),
  ("Who was King Henry VIII?", "English king with six wives."),
  ("Who was Mary Queen of Scots?", "Scottish queen, rival to Elizabeth I."),
  ("Who was Louis XIV?", "French Sun King."),
  ("Who was Catherine the Great?", "Russian empress."),
  ("Who was Peter the Great?", "Modernizer of Russia."),
  ("Who was Tutankhamun?", "Boy king of ancient Egypt."),
  ("Who was Ramesses II?", "Great pharaoh of Egypt."),
  ("Who was Helen of Troy?", "Legendary cause of Trojan War."),
  ("Who was Homer?", "Greek poet of Iliad and Odyssey."),
  ("Who was Virgil?", "Roman poet of the Aeneid."),
  ("Who was Dante?", "Italian poet of the Divine Comedy."),
  ("Who was Galileo's rival church?", "The Catholic Church."),
  ("Who was Pythagoras?", "Greek mathematician."),
  ("Who was Euclid?", "Father of geometry."),
  ("Who was Archimedes?", "Greek mathematician and inventor."),
  ("Who was Hippocrates?", "Father of medicine."),
  ("Who was Sigmund Freud?", "Founder of psychoanalysis."),
  ("Who was Carl Jung?", "Psychologist of archetypes."),
]

def gen_history_figures():
  items = 
  for q, a in HIST_FIGURES:
  items.append({"category": "history_figures", "q": q, "a": a})
  # 76件 → 150件
  for q, a in HIST_FIGURES:
  q2 = "Tell me about: " + q.split("Who was ", 1)[-1].replace("?", ".")
  items.append({"category": "history_figures", "q": q2, "a": a})
  write_jsonl("history_figures", items[:150])

if __name__ == "__main__":
  gen_science_basics()
  gen_science_animals()
  gen_science_planets()
  gen_science_human_body()
  gen_history_events()
  gen_history_figures()
