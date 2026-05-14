# 🐕 CeREbrus
### Enterprise Retail Intelligence Platform
**lablab.ai Hackathon — Track 2: AI Agents with Google AI Studio**
> *"Cerberus in mythology had three heads guarding the gate. CeREbrus has three agents guarding your enterprise data."*

---

## The Problem
Enterprise retail teams are split between two pain points simultaneously:

- **Ops & buyers** can't find answers buried across hundreds of policy docs, SOPs, and vendor contracts
- **Customer-facing reps** switch between 3–4 tools just to get context before a single customer interaction

Both problems cost enterprises real money in wasted time and bad decisions.

## The Solution
CeREbrus is a multi-agent RAG system with **three specialist heads** — one interface, two enterprise problems solved.

| Head | Agent | Does |
|---|---|---|
| 🧠 Head 1 | **Knowledge Guardian** | Answers policy, SOP, vendor contract, and compliance questions |
| 👤 Head 2 | **Customer Intel** | Retrieves full customer profiles — purchases, tickets, tier, rep notes |
| 📋 Head 3 | **Synthesis** | Generates pre-interaction briefs combining both sources |

---

## Setup (5 minutes)

### 1. Create virtual environment
```bash
cd cerebrus
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key
- Go to https://aistudio.google.com → Get API Key (free, no credit card)
- Open `.env` → replace `your_api_key_here` with your key

### 4. Run
```bash
# Browser UI — best for demos
adk web cerebrus

# Terminal mode
adk run cerebrus
```

---

## Demo Script (show this to judges)

**Knowledge queries:**
```
"What's our return policy for items bought during a promotional event?"
"When does our TechSupply Co. vendor contract renew?"
"What's the escalation procedure for a $750 refund request?"
"What are our PCI compliance rules around storing card data?"
```

**Customer intelligence:**
```
"Pull up customer 4821"
"What's going on with customer 7710?"
"Give me the profile for customer 2034"
```

**Pre-interaction briefs:**
```
"Brief me on customer 4821 before my support call"
"I'm about to make a sales call to customer 2034 — what should I know?"
"Retention brief for customer 7710"
```

---

## Project Structure
```
cerebrus/
├── agent.py           ← All three agents + tools + mock data (start here)
├── __init__.py
├── requirements.txt
├── .env               ← Your API key (never commit this)
└── README.md
```

## Track Alignment
- ✅ Multi-agent system using Gemini (three specialist sub-agents)
- ✅ Internal AI tools / knowledge base ops
- ✅ Enterprise integrations (CRM + document RAG)
- ✅ Agent-driven workflows responding to user input and context
- ✅ Working prototype with demo-ready UI via `adk web`

---

## Pitch Tagline
**"Three heads. One source of truth."**
