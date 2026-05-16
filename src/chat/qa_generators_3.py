"""
Q&A 生成器 3: literature_*, shakespeare, language_*
"""

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "qa"
OUT_DIR.mkdir(exist_ok=True)

random.seed(44)

def write_jsonl(category: str, items: list):
  path = OUT_DIR / f"{category}.jsonl"
  with path.open('w', encoding='utf-8') as f:
  for it in items:
  f.write(json.dumps(it, ensure_ascii=False) + '\n')
  print(f"[{category}] {len(items)} pairs -> {path}")

# ---------------------------------------------------------------------------
# literature_authors
# ---------------------------------------------------------------------------

AUTHORS = [
  ("William Shakespeare", "Romeo and Juliet"),
  ("Jane Austen", "Pride and Prejudice"),
  ("Charles Dickens", "Oliver Twist"),
  ("Mark Twain", "Tom Sawyer"),
  ("Ernest Hemingway", "The Old Man and the Sea"),
  ("F. Scott Fitzgerald", "The Great Gatsby"),
  ("Leo Tolstoy", "War and Peace"),
  ("Fyodor Dostoevsky", "Crime and Punishment"),
  ("Victor Hugo", "Les Miserables"),
  ("Alexandre Dumas", "The Three Musketeers"),
  ("Miguel de Cervantes", "Don Quixote"),
  ("Homer", "The Iliad"),
  ("Virgil", "The Aeneid"),
  ("Dante Alighieri", "The Divine Comedy"),
  ("J.R.R. Tolkien", "The Lord of the Rings"),
  ("C.S. Lewis", "The Chronicles of Narnia"),
  ("J.K. Rowling", "Harry Potter"),
  ("George Orwell", "1984"),
  ("Aldous Huxley", "Brave New World"),
  ("Ray Bradbury", "Fahrenheit 451"),
  ("Isaac Asimov", "Foundation"),
  ("Arthur C. Clarke", "2001 A Space Odyssey"),
  ("Stephen King", "It"),
  ("Agatha Christie", "Murder on the Orient Express"),
  ("Arthur Conan Doyle", "Sherlock Holmes"),
  ("Edgar Allan Poe", "The Raven"),
  ("Herman Melville", "Moby Dick"),
  ("Mary Shelley", "Frankenstein"),
  ("Bram Stoker", "Dracula"),
  ("Oscar Wilde", "The Picture of Dorian Gray"),
  ("Lewis Carroll", "Alice in Wonderland"),
  ("Robert Louis Stevenson", "Treasure Island"),
  ("Jules Verne", "Around the World in 80 Days"),
  ("H.G. Wells", "The Time Machine"),
  ("Franz Kafka", "The Metamorphosis"),
  ("James Joyce", "Ulysses"),
  ("Virginia Woolf", "Mrs Dalloway"),
  ("Marcel Proust", "In Search of Lost Time"),
  ("Gabriel Garcia Marquez", "One Hundred Years of Solitude"),
  ("Albert Camus", "The Stranger"),
  ("Jean-Paul Sartre", "Nausea"),
  ("Haruki Murakami", "Norwegian Wood"),
  ("Yukio Mishima", "Confessions of a Mask"),
  ("Kazuo Ishiguro", "The Remains of the Day"),
  ("Margaret Atwood", "The Handmaid's Tale"),
  ("Toni Morrison", "Beloved"),
  ("Maya Angelou", "I Know Why the Caged Bird Sings"),
  ("Harper Lee", "To Kill a Mockingbird"),
  ("John Steinbeck", "The Grapes of Wrath"),
  ("William Faulkner", "The Sound and the Fury"),
]

AUTHOR_Q_FORMS = [
  "Who wrote {book}?",
  "Author of {book}?",
  "Who is the author of {book}?",
  "Name the author of {book}.",
]
AUTHOR_A_FORMS = [
  "{author} wrote it.",
  "It was {author}.",
  "{author}.",
  "{book} was written by {author}.",
]
WORK_Q_FORMS = [
  "Name a work by {author}.",
  "What did {author} write?",
  "Tell me a book by {author}.",
]
WORK_A_FORMS = [
  "{author} wrote {book}.",
  "{book}.",
  "One famous work is {book}.",
]

def gen_literature_authors():
  items = 
  for author, book in AUTHORS:
  q = random.choice(AUTHOR_Q_FORMS).format(book=book)
  a = random.choice(AUTHOR_A_FORMS).format(author=author, book=book)
  items.append({"category": "literature_authors", "q": q, "a": a})
  for author, book in AUTHORS:
  q = random.choice(WORK_Q_FORMS).format(author=author)
  a = random.choice(WORK_A_FORMS).format(author=author, book=book)
  items.append({"category": "literature_authors", "q": q, "a": a})
  random.shuffle(items)
  write_jsonl("literature_authors", items[:100])

# ---------------------------------------------------------------------------
# literature_quotes
# ---------------------------------------------------------------------------

QUOTES = [
  ("To be or not to be.", "Shakespeare, Hamlet"),
  ("All the world's a stage.", "Shakespeare, As You Like It"),
  ("Brevity is the soul of wit.", "Shakespeare, Hamlet"),
  ("Et tu, Brute?", "Shakespeare, Julius Caesar"),
  ("To thine own self be true.", "Shakespeare, Hamlet"),
  ("Some are born great.", "Shakespeare, Twelfth Night"),
  ("I think, therefore I am.", "Rene Descartes"),
  ("Knowledge is power.", "Francis Bacon"),
  ("To err is human.", "Alexander Pope"),
  ("Veni, vidi, vici.", "Julius Caesar"),
  ("Carpe diem.", "Horace"),
  ("Cogito, ergo sum.", "Rene Descartes"),
  ("All for one and one for all.", "Alexandre Dumas"),
  ("It was the best of times.", "Charles Dickens"),
  ("It is a truth universally acknowledged.", "Jane Austen"),
  ("Call me Ishmael.", "Herman Melville, Moby Dick"),
  ("Big Brother is watching you.", "George Orwell, 1984"),
  ("So it goes.", "Kurt Vonnegut, Slaughterhouse-Five"),
  ("All happy families are alike.", "Leo Tolstoy"),
  ("Whatever you are, be a good one.", "Abraham Lincoln"),
  ("I have a dream.", "Martin Luther King Jr."),
  ("Ask not what your country can do for you.", "John F. Kennedy"),
  ("The only thing we have to fear is fear itself.", "F.D. Roosevelt"),
  ("Imagination is more important than knowledge.", "Albert Einstein"),
  ("Be the change you wish to see.", "Mahatma Gandhi"),
  ("Life is what happens when you make plans.", "John Lennon"),
  ("Stay hungry, stay foolish.", "Steve Jobs"),
  ("Float like a butterfly, sting like a bee.", "Muhammad Ali"),
  ("I came, I saw, I conquered.", "Julius Caesar"),
  ("Give me liberty, or give me death.", "Patrick Henry"),
  ("The pen is mightier than the sword.", "Edward Bulwer-Lytton"),
  ("A journey of a thousand miles begins.", "Lao Tzu"),
  ("Curiosity killed the cat.", "old saying"),
  ("Practice makes perfect.", "old saying"),
  ("Time is money.", "Benjamin Franklin"),
  ("Eureka.", "Archimedes"),
  ("That's one small step for man.", "Neil Armstrong"),
  ("Hasta la vista, baby.", "Terminator 2 movie"),
  ("May the Force be with you.", "Star Wars"),
  ("To infinity and beyond.", "Toy Story"),
]

def gen_literature_quotes():
  items = 
  for q, src in QUOTES:
  items.append({"category": "literature_quotes",
  "q": f"Who said: {q}?",
  "a": f"That was {src}."})
  items.append({"category": "literature_quotes",
  "q": f"Tell me a quote from {src.split(',')[0]}.",
  "a": f"'{q}'"})
  write_jsonl("literature_quotes", items[:80])

# ---------------------------------------------------------------------------
# shakespeare
# ---------------------------------------------------------------------------

SHAKESPEARE_FACTS = [
  ("Who wrote Hamlet?", "William Shakespeare."),
  ("Who wrote Romeo and Juliet?", "William Shakespeare."),
  ("Who wrote Macbeth?", "William Shakespeare."),
  ("Who wrote King Lear?", "William Shakespeare."),
  ("Who wrote Othello?", "William Shakespeare."),
  ("Who wrote Julius Caesar?", "William Shakespeare."),
  ("Who wrote The Tempest?", "William Shakespeare."),
  ("Who wrote A Midsummer Night's Dream?", "William Shakespeare."),
  ("Who is Hamlet's father?", "The ghost of the old king."),
  ("Who is Hamlet's mother?", "Queen Gertrude."),
  ("Who is Hamlet's love?", "Ophelia."),
  ("Who is Hamlet's uncle?", "Claudius, the new king."),
  ("Who killed Hamlet's father?", "Claudius poisoned him."),
  ("Who is Romeo's love?", "Juliet."),
  ("What family is Romeo from?", "House of Montague."),
  ("What family is Juliet from?", "House of Capulet."),
  ("Where is Romeo and Juliet set?", "In Verona, Italy."),
  ("How does Romeo die?", "He drinks poison."),
  ("How does Juliet die?", "She stabs herself."),
  ("Who is Macbeth's wife?", "Lady Macbeth."),
  ("Who do the witches greet?", "Macbeth and Banquo."),
  ("How does Macbeth die?", "Macduff slays him."),
  ("Who is King Lear's daughters?", "Goneril, Regan, and Cordelia."),
  ("Which daughter loves Lear?", "Cordelia, the youngest."),
  ("Who is Othello's wife?", "Desdemona."),
  ("Who deceives Othello?", "Iago, his ensign."),
  ("Who kills Caesar?", "Brutus and other senators."),
  ("What does Brutus say?", "He says 'Et tu, Brute?'"),
  ("Wait, who said Et tu, Brute?", "Julius Caesar said it."),
  ("Who is Prospero?", "The exiled duke in The Tempest."),
  ("Who is Ariel?", "A spirit serving Prospero."),
  ("Who is Caliban?", "A creature in The Tempest."),
  ("Who is Puck?", "A trickster fairy in Midsummer."),
  ("Who is Oberon?", "King of the fairies."),
  ("Who is Titania?", "Queen of the fairies."),
  ("Where was Shakespeare born?", "In Stratford-upon-Avon."),
  ("When was Shakespeare born?", "In 1564."),
  ("When did Shakespeare die?", "In 1616."),
  ("How old was Shakespeare when he died?", "52 years old."),
  ("What was Shakespeare's theater called?", "The Globe Theater."),
  ("What is a sonnet?", "A 14-line poem."),
  ("How many sonnets did Shakespeare write?", "154 sonnets."),
  ("What is iambic pentameter?", "A meter of ten beats per line."),
  ("Did Shakespeare invent words?", "Yes, hundreds of them."),
  ("Did Shakespeare invent eyeball?", "He likely coined eyeball."),
  ("What is 'Wherefore'?", "It means 'why' in old English."),
  ("What does 'thee' mean?", "It means 'you' in old English."),
  ("What does 'thou' mean?", "Also 'you' in old English."),
  ("What does 'art' mean here?", "'are', as in 'thou art'."),
  ("What is a soliloquy?", "A speech alone on stage."),
  ("Name a Shakespeare comedy.", "A Midsummer Night's Dream."),
  ("Name a Shakespeare tragedy.", "Hamlet."),
  ("Name a Shakespeare history.", "Henry V."),
  ("Who is Falstaff?", "A jolly knight in Henry IV."),
  ("Who is Iago?", "Villain of Othello."),
  ("Who is Cordelia?", "King Lear's loyal daughter."),
  ("Who is Lady Macbeth?", "Macbeth's ambitious wife."),
  ("Who is Bottom?", "A weaver in Midsummer."),
  ("Who is Shylock?", "A character in Merchant of Venice."),
  ("Who is Mercutio?", "Romeo's witty friend."),
  ("Who is Tybalt?", "Juliet's cousin who kills Mercutio."),
  ("Who is Friar Lawrence?", "Priest who marries Romeo and Juliet."),
  ("Who is Nurse?", "Juliet's caretaker."),
  ("Who is Polonius?", "Ophelia's father in Hamlet."),
  ("Who is Laertes?", "Ophelia's brother in Hamlet."),
  ("Who is Horatio?", "Hamlet's loyal friend."),
  ("Who killed Polonius?", "Hamlet did, by mistake."),
  ("Who is Banquo?", "Macbeth's friend and rival."),
  ("Who is Duncan?", "King slain by Macbeth."),
]

def gen_shakespeare():
  items = 
  for q, a in SHAKESPEARE_FACTS:
  items.append({"category": "shakespeare", "q": q, "a": a})
  write_jsonl("shakespeare", items[:100])

# ---------------------------------------------------------------------------
# language_definitions
# ---------------------------------------------------------------------------

WORD_DEFS = [
  ("ephemeral", "lasting for a very short time"),
  ("ubiquitous", "present everywhere"),
  ("serendipity", "happy accident"),
  ("melancholy", "deep sadness"),
  ("eloquent", "speaking well"),
  ("perpetual", "continuing forever"),
  ("ambiguous", "having more than one meaning"),
  ("benevolent", "kind and generous"),
  ("candid", "honest and direct"),
  ("diligent", "hard working"),
  ("eccentric", "unusual or odd"),
  ("frugal", "careful with money"),
  ("garrulous", "talking too much"),
  ("haughty", "proud and arrogant"),
  ("indolent", "lazy"),
  ("jovial", "cheerful and friendly"),
  ("keen", "eager or sharp"),
  ("loquacious", "talkative"),
  ("malice", "wish to harm"),
  ("naive", "lacking experience"),
  ("obstinate", "stubborn"),
  ("pristine", "clean and untouched"),
  ("quaint", "old-fashioned charm"),
  ("resilient", "able to recover quickly"),
  ("sagacious", "wise"),
  ("tenacious", "holding on firmly"),
  ("urbane", "polite and refined"),
  ("verbose", "using too many words"),
  ("wary", "cautious"),
  ("xenophobic", "fearing foreigners"),
  ("yearn", "to long for"),
  ("zealous", "showing great energy"),
  ("abundant", "plentiful"),
  ("brisk", "quick and lively"),
  ("clandestine", "secret"),
  ("dormant", "inactive but capable"),
  ("emanate", "to flow out from"),
  ("furtive", "secret and sly"),
  ("grandiose", "impressive in size"),
  ("hapless", "unfortunate"),
  ("imminent", "about to happen"),
  ("judicious", "wise in judgment"),
  ("kindle", "to ignite"),
  ("languid", "weak or slow"),
  ("meticulous", "very careful"),
  ("notorious", "famous for bad reasons"),
  ("opaque", "not see-through"),
  ("placid", "calm"),
  ("quell", "to suppress"),
  ("revere", "to honor deeply"),
  ("scrutinize", "to examine carefully"),
  ("trepidation", "fear"),
  ("ubiquitous", "everywhere at once"),
  ("vigilant", "watchful"),
  ("whimsical", "playful or fanciful"),
  ("yielding", "giving way"),
  ("zenith", "highest point"),
  ("aberration", "departure from normal"),
  ("benign", "harmless"),
  ("colossal", "huge"),
  ("dubious", "doubtful"),
  ("enigma", "a mystery"),
  ("fervent", "passionate"),
  ("gregarious", "social"),
  ("hapless", "unlucky"),
  ("inane", "silly or empty"),
  ("juxtapose", "to place side by side"),
  ("knack", "natural skill"),
  ("lethargy", "tiredness"),
  ("mundane", "ordinary"),
  ("nuance", "subtle distinction"),
  ("obsolete", "out of date"),
  ("pensive", "deeply thoughtful"),
  ("quintessential", "the most typical"),
  ("ruse", "trick"),
  ("solitude", "being alone"),
  ("taciturn", "quiet by nature"),
  ("vex", "to annoy"),
  ("wane", "to decrease"),
  ("yore", "the past"),
  ("zest", "lively enjoyment"),
  ("acquiesce", "to accept reluctantly"),
  ("brusque", "abrupt"),
  ("cordial", "warm and friendly"),
  ("debacle", "complete failure"),
  ("ebullient", "cheerfully bubbly"),
  ("facetious", "joking awkwardly"),
  ("germane", "relevant"),
  ("hubris", "excessive pride"),
  ("impasse", "deadlock"),
  ("juvenile", "youthful"),
  ("laconic", "using few words"),
  ("mercurial", "changing moods"),
  ("nadir", "lowest point"),
  ("opulent", "richly luxurious"),
  ("pugnacious", "eager to fight"),
  ("quagmire", "difficult situation"),
  ("recalcitrant", "stubbornly resistant"),
  ("supercilious", "snobbishly proud"),
  ("turbid", "muddy or unclear"),
  ("voracious", "very hungry"),
  ("watershed", "turning point"),
]

def gen_language_definitions():
  items = 
  for word, defn in WORD_DEFS:
  items.append({"category": "language_definitions",
  "q": f"What does {word} mean?",
  "a": f"It means {defn}."})
  for word, defn in WORD_DEFS:
  items.append({"category": "language_definitions",
  "q": f"Define {word}.",
  "a": f"It is {defn}."})
  write_jsonl("language_definitions", items[:200])

# ---------------------------------------------------------------------------
# language_synonyms
# ---------------------------------------------------------------------------

SYNONYMS = [
  ("happy", "joyful"), ("sad", "unhappy"), ("big", "large"),
  ("small", "tiny"), ("fast", "quick"), ("slow", "sluggish"),
  ("smart", "clever"), ("strong", "powerful"), ("weak", "feeble"),
  ("beautiful", "gorgeous"), ("ugly", "hideous"), ("hot", "warm"),
  ("cold", "chilly"), ("dark", "gloomy"), ("light", "bright"),
  ("rich", "wealthy"), ("poor", "needy"), ("brave", "courageous"),
  ("afraid", "scared"), ("angry", "furious"), ("kind", "gentle"),
  ("mean", "cruel"), ("funny", "humorous"), ("boring", "dull"),
  ("interesting", "fascinating"), ("difficult", "hard"), ("easy", "simple"),
  ("clean", "tidy"), ("dirty", "filthy"), ("loud", "noisy"),
  ("quiet", "silent"), ("new", "fresh"), ("old", "ancient"),
  ("young", "youthful"), ("clever", "smart"), ("dumb", "foolish"),
  ("sleep", "slumber"), ("eat", "consume"), ("drink", "sip"),
  ("walk", "stroll"), ("run", "sprint"), ("look", "gaze"),
  ("speak", "talk"), ("laugh", "chuckle"), ("cry", "weep"),
  ("buy", "purchase"), ("sell", "vend"), ("begin", "start"),
  ("end", "finish"), ("help", "assist"), ("teach", "instruct"),
]

def gen_language_synonyms():
  items = 
  for w1, w2 in SYNONYMS:
  items.append({"category": "language_synonyms",
  "q": f"What is a synonym for {w1}?",
  "a": f"A synonym is {w2}."})
  for w1, w2 in SYNONYMS:
  items.append({"category": "language_synonyms",
  "q": f"Synonym of {w1}?",
  "a": f"{w2}."})
  write_jsonl("language_synonyms", items[:100])

# ---------------------------------------------------------------------------
# language_antonyms
# ---------------------------------------------------------------------------

ANTONYMS = [
  ("hot", "cold"), ("big", "small"), ("fast", "slow"),
  ("good", "bad"), ("happy", "sad"), ("rich", "poor"),
  ("strong", "weak"), ("light", "dark"), ("up", "down"),
  ("left", "right"), ("in", "out"), ("yes", "no"),
  ("open", "closed"), ("clean", "dirty"), ("full", "empty"),
  ("easy", "hard"), ("near", "far"), ("early", "late"),
  ("young", "old"), ("new", "old"), ("alive", "dead"),
  ("true", "false"), ("right", "wrong"), ("first", "last"),
  ("noisy", "quiet"), ("sweet", "sour"), ("kind", "cruel"),
  ("smart", "dumb"), ("beautiful", "ugly"), ("front", "back"),
  ("inside", "outside"), ("above", "below"), ("more", "less"),
  ("buy", "sell"), ("win", "lose"), ("love", "hate"),
  ("laugh", "cry"), ("push", "pull"), ("give", "take"),
  ("rise", "fall"),
]

def gen_language_antonyms():
  items = 
  for w1, w2 in ANTONYMS:
  items.append({"category": "language_antonyms",
  "q": f"What is the opposite of {w1}?",
  "a": f"The opposite is {w2}."})
  for w1, w2 in ANTONYMS:
  items.append({"category": "language_antonyms",
  "q": f"Antonym of {w1}?",
  "a": f"{w2}."})
  write_jsonl("language_antonyms", items[:80])

# ---------------------------------------------------------------------------
# language_grammar
# ---------------------------------------------------------------------------

GRAMMAR_QA = [
  ("What is a noun?", "A person, place, or thing."),
  ("What is a verb?", "A word for an action."),
  ("What is an adjective?", "A word that describes a noun."),
  ("What is an adverb?", "A word that describes a verb."),
  ("What is a pronoun?", "A word that stands for a noun."),
  ("What is a preposition?", "A word like in, on, at."),
  ("What is a conjunction?", "A word like and, but, or."),
  ("What is an article?", "A word like a, an, the."),
  ("What is a sentence?", "A group of words with meaning."),
  ("What is a paragraph?", "A group of related sentences."),
  ("What is punctuation?", "Marks like periods and commas."),
  ("What is a comma?", "A pause mark: ,"),
  ("What is a period?", "An end mark for sentences: ."),
  ("What is a question mark?", "End mark for questions: ?"),
  ("What is an apostrophe?", "Mark for possession: '"),
  ("What is a colon?", "Two dots that introduce: :"),
  ("What is a semicolon?", "Used to link clauses: ;"),
  ("What is a subject?", "What the sentence is about."),
  ("What is a predicate?", "What the subject does."),
  ("What is an object?", "What receives the action."),
  ("What is a clause?", "A group of words with a verb."),
  ("What is a phrase?", "A group of words, no verb."),
  ("What is a synonym?", "A word with the same meaning."),
  ("What is an antonym?", "A word with the opposite meaning."),
  ("What is a homonym?", "A word with multiple meanings."),
  ("What is a homophone?", "A word sounding like another."),
  ("Give an example of a noun.", "Dog, city, book are nouns."),
  ("Give an example of a verb.", "Run, eat, sleep are verbs."),
  ("Give an example of an adjective.", "Red, tall, quick."),
  ("Give an example of an adverb.", "Quickly, well, often."),
  ("What is plural?", "More than one of something."),
  ("What is singular?", "Just one of something."),
  ("What is past tense?", "Action that already happened."),
  ("What is present tense?", "Action happening now."),
  ("What is future tense?", "Action that will happen."),
  ("What is a vowel?", "A, E, I, O, U sounds."),
  ("What is a consonant?", "Letters not vowels."),
  ("How many letters in the alphabet?", "26 letters."),
  ("How many vowels?", "Five main vowels."),
  ("What is a syllable?", "A unit of pronunciation."),
]

def gen_language_grammar():
  items = 
  for q, a in GRAMMAR_QA:
  items.append({"category": "language_grammar", "q": q, "a": a})
  for q, a in GRAMMAR_QA:
  items.append({"category": "language_grammar",
  "q": "Grammar: " + q.lower(),
  "a": a})
  write_jsonl("language_grammar", items[:80])

if __name__ == "__main__":
  gen_literature_authors()
  gen_literature_quotes()
  gen_shakespeare()
  gen_language_definitions()
  gen_language_synonyms()
  gen_language_antonyms()
  gen_language_grammar()
