# Value Seat  24/7 Restaurant Support Chatbot (English + Arabic)

A chatbot that answers customer questions about a restaurant menu, ingredients, allergens, and diets, in multilingual **English and Arabic**. It runs fully **locally**: a small language model (Qwen2.5:3b through Ollama) plus a vector database (ChromaDB). No cloud.



## Why I built it

i wanted to create an customer self service chatbot that can answer basic and simple question that usually needs a phone call and to be on the waiting list just to get an simple answer , and it answers all type of questions that its related to the restaurant and is the meal prep is done. 

## How it works

```
Customer question
      |
      v
Fixed-reply check  --(emergency / complaint / prompt-injection)--> fixed safe reply
      |
      v
Detect diet + allergies (buttons on the page + words in the message, English & Arabic)
      |
      v
Search the menu in ChromaDB (multilingual embeddings)
      |
      v
Python filters the dishes (category, food type, price, diet, allergens)
and marks each dish  SAFE / NOT SAFE  for this customer
      |
      v
Prompt = restaurant info + checked dishes + customer needs + reply language
      |
      v
Qwen2.5:3b (Ollama, temperature 0.2)  -->  answer in English or Arabic
      |
      v
Output check (no leaked instructions) --> reply
```

### Main features

- **Bilingual:** Detects Arabic in the message and replies in Arabic. The menu has Arabic dish names.
- **Allergen and diet safety in code:** `problems()` checks each dish against the customer's needs. "May contain" counts as **not safe**.
- **Preference buttons** on the page: Vegetarian, Vegan, Halal, Gluten-free, Dairy-free, Nut allergy, Egg allergy, Shellfish allergy. The bot also remembers things like "I am allergic to ..." during the chat.
- **Semantic search (RAG):** Menu and info entries are embedded with `paraphrase-multilingual-MiniLM-L12-v2` and stored in ChromaDB. The database rebuilds by itself when the JSON file changes (MD5 hash check).
- **Follow-up questions:** Short messages like "yes please" are searched together with the previous question.
- **Guardrails:**
  - Emergency words (for example, allergic reaction) get a fixed reply that sends the customer to emergency help.
  - Complaints get a fixed reply that sends the customer to the restaurant manager and official channels.
  - Prompt-injection attempts ("ignore your instructions") get a fixed reply.
  - The bot does not share recipes, cannot take orders or payments, and never calls a dish "halal certified" unless the restaurant info says so.
  - If the model output contains internal prompt text, it is replaced by a safe fallback.
  - Messages longer than 1000 characters are rejected.
- **Session memory:** Short conversation history (last 6 messages) per session.

## Data

`restaurant_data.json` holds a **sample menu of 85 dishes** in 9 categories. Each dish has: name (English + Arabic), ingredients, price, dietary tags, allergens it contains, allergens it may contain, pork/alcohol flags, and a search text.

> The allergen information is **sample data for this demo**. Do not use it as real allergy advice.

## Tech stack

Python · Flask · ChromaDB · sentence-transformers · Ollama (Qwen2.5:3b) · HTML

## Project structure

```
.
├── app.py                  # Flask server + chatbot logic
├── restaurant_data.json    # Menu (85 dishes) and restaurant info entries
├── templates/page.html     # Chat web page
├── static/                 # Images
├── requirements.txt
└── restaurant_db/          # Created automatically on first run (not in repo)
```

## How to run

**1. Install the libraries**

```bash
pip install -r requirements.txt
```

**2. Install Ollama and the model**

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b
```

**3. Start the server**

```bash
python app.py
```

The browser opens at `http://127.0.0.1:5000`. The first run downloads the embedding model from Hugging Face (internet needed one time) and builds the database.

To use another Ollama model:

```bash
# Windows PowerShell
$env:OLLAMA_MODEL = "qwen2.5:1.5b"; python app.py
```

## API

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Chat page |
| `/start_session` | POST | Creates a session id |
| `/chat` | POST | Body: `{ "session_id", "query", "prefs": [] }` -> `{ "reply", "session_id" }` |
| `/api/session/end` | POST | Deletes the session memory |

## Limitations (honest list)

- **No formal evaluation yet.** I tested it by hand, but I have no accuracy numbers.
- **Only 3 of 14 restaurant info entries are finished** (the others are marked `todo` and are ignored). Questions about delivery, parking, reservations, or payment may get "ask the staff".
- Diet and allergy detection uses **word lists** (English and Arabic). Unusual wording can be missed.
- A 3B model can still make mistakes in wording, even when the facts are correct.
- Sessions are stored in memory, so they are lost when the server restarts.
- It cannot take orders or payments.
- Menu and allergen data are sample data, not verified by a real kitchen.

## Future work

- Write a test set of questions (allergy, diet, Arabic, prompt injection) and measure the results.
- Finish the remaining restaurant info entries.
- Add a real database for sessions and a simple admin page to edit the menu.
- Add voice input.

## Author

**Jassim** — Artificial Intelligence Engineering student, Near East University.
LinkedIn: [add your link here] · GitHub: [add your link here]

## License

MIT (see `LICENSE`).
