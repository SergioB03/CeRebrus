"""
CeREbrus — Enterprise Retail Intelligence Platform
lablab.ai Hackathon | Track 2: AI Agents with Google AI Studio
Powered by Google ADK + Gemini 2.5 Flash

"Cerberus in mythology had three heads guarding the gate.
 CeREbrus has three agents guarding your enterprise data."

Architecture:
  Orchestrator (CeREbrus)
  ├── Head 1: Knowledge Base Agent   — RAG over internal policy docs, SOPs, vendor contracts
  ├── Head 2: Customer Intel Agent   — purchase history, support tickets, behavior patterns
  └── Head 3: Summary Agent          — pre-interaction briefs, cross-source synthesis
"""

import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent

load_dotenv()

# ─────────────────────────────────────────────
# MOCK DATA LAYER
# In production: replace with real vector DB (Pinecone, Weaviate)
# and CRM API (Salesforce, HubSpot) calls
# ─────────────────────────────────────────────

POLICY_DOCS = {
    "return_policy": {
        "title": "Return & Refund Policy v2.3",
        "content": (
            "Standard return window is 30 days with receipt. Electronics have a 15-day return window. "
            "Items purchased during promotional events are eligible for store credit only, not cash refunds. "
            "Opened software, digital downloads, and personalized items are non-returnable. "
            "Manager override required for returns exceeding $500."
        ),
        "last_updated": "2025-11-01"
    },
    "vendor_terms": {
        "title": "Vendor Agreement — TechSupply Co. (Contract #VS-2024-089)",
        "content": (
            "Net-30 payment terms. Minimum order quantity: 500 units per SKU. "
            "Price lock guaranteed through Q3 2026. Penalty clause: 2% per week for late payments beyond 45 days. "
            "Renewal date: September 1, 2026. Exclusive distribution rights for Southeast region. "
            "SLA: 98% on-time delivery or vendor absorbs shipping cost differential."
        ),
        "last_updated": "2024-09-01"
    },
    "employee_escalation": {
        "title": "Customer Escalation SOP v1.8",
        "content": (
            "Tier 1: Front-line rep handles complaints under $200. "
            "Tier 2: Supervisor approval required for refunds $200–$999. "
            "Tier 3: District manager sign-off for refunds over $1,000 or repeat escalations (3+ contacts). "
            "All escalations must be logged in CRM within 2 hours. "
            "VIP customers (Gold/Platinum tier) skip Tier 1 — route directly to Tier 2."
        ),
        "last_updated": "2025-06-15"
    },
    "compliance": {
        "title": "Data Privacy & PCI Compliance Guidelines",
        "content": (
            "Never store full credit card numbers in CRM notes. Last 4 digits only. "
            "Customer data requests (CCPA/GDPR) must be fulfilled within 30 days. "
            "Screen recordings of customer interactions are retained for 90 days. "
            "PCI DSS Level 1 compliance required for all payment integrations."
        ),
        "last_updated": "2025-03-20"
    }
}

CUSTOMER_DB = {
    "4821": {
        "name": "Marcus Johnson",
        "tier": "Gold",
        "lifetime_value": 14820.50,
        "recent_purchases": [
            {"item": "65-inch OLED TV", "amount": 1899.99, "date": "2026-04-10", "status": "delivered"},
            {"item": "Soundbar Pro X", "amount": 349.99, "date": "2026-03-22", "status": "delivered"},
            {"item": "Extended Warranty Pack", "amount": 199.00, "date": "2026-04-10", "status": "active"},
        ],
        "open_tickets": [
            {"id": "TK-9921", "issue": "TV remote not pairing", "status": "open", "created": "2026-04-18"},
        ],
        "notes": "Loyal customer since 2019. Prefers email communication. Sensitive about wait times."
    },
    "2034": {
        "name": "Sofia Reyes",
        "tier": "Platinum",
        "lifetime_value": 38400.00,
        "recent_purchases": [
            {"item": "MacBook Pro 16\"", "amount": 2499.00, "date": "2026-05-01", "status": "delivered"},
            {"item": "USB-C Hub Bundle", "amount": 89.99, "date": "2026-05-01", "status": "delivered"},
        ],
        "open_tickets": [],
        "notes": "Executive buyer for a mid-size firm. Bulk purchaser. Contract pricing eligible."
    },
    "7710": {
        "name": "Derek Park",
        "tier": "Standard",
        "lifetime_value": 1240.00,
        "recent_purchases": [
            {"item": "Gaming Headset Z", "amount": 129.99, "date": "2026-04-28", "status": "delivered"},
        ],
        "open_tickets": [
            {"id": "TK-9988", "issue": "Headset defective — no audio in left ear", "status": "escalated", "created": "2026-05-02"},
            {"id": "TK-9901", "issue": "Return request denied — outside window", "status": "closed", "created": "2026-04-15"},
        ],
        "notes": "Two escalations in 30 days. Frustrated customer — handle with care."
    }
}

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def search_knowledge_base(query: str) -> dict:
    """
    Searches internal enterprise knowledge base for policies,
    SOPs, vendor contracts, and compliance docs.

    Args:
        query: Natural language question from an employee

    Returns:
        Matching document title, relevant content, and last updated date
    """
    q = query.lower()

    # Keyword routing — in production this is a vector similarity search
    if any(w in q for w in ["return", "refund", "exchange", "promo", "promotional"]):
        doc = POLICY_DOCS["return_policy"]
    elif any(w in q for w in ["vendor", "supplier", "contract", "payment", "techsupply", "renewal", "penalty"]):
        doc = POLICY_DOCS["vendor_terms"]
    elif any(w in q for w in ["escalat", "tier", "supervisor", "manager", "complaint", "override"]):
        doc = POLICY_DOCS["employee_escalation"]
    elif any(w in q for w in ["privacy", "pci", "gdpr", "ccpa", "compliance", "data", "credit card"]):
        doc = POLICY_DOCS["compliance"]
    else:
        return {
            "found": False,
            "message": "No matching policy found. Try rephrasing or contact your ops manager.",
            "suggestion": "Available topics: returns, vendor contracts, escalation procedures, compliance"
        }

    return {
        "found": True,
        "document": doc["title"],
        "content": doc["content"],
        "last_updated": doc["last_updated"],
        "source": "CeREbrus Internal Knowledge Base"
    }


def get_customer_profile(customer_id: str) -> dict:
    """
    Retrieves a full customer intelligence profile including
    purchase history, open support tickets, tier status, and rep notes.

    Args:
        customer_id: The customer's ID number (string)

    Returns:
        Complete customer profile with context for pre-interaction briefing
    """
    customer = CUSTOMER_DB.get(customer_id)

    if not customer:
        return {
            "found": False,
            "message": f"No customer found with ID {customer_id}. Verify the ID and try again."
        }

    open_ticket_count = len(customer["open_tickets"])
    recent_spend = sum(p["amount"] for p in customer["recent_purchases"])

    return {
        "found": True,
        "customer_id": customer_id,
        "name": customer["name"],
        "tier": customer["tier"],
        "lifetime_value": f"${customer['lifetime_value']:,.2f}",
        "recent_spend_90_days": f"${recent_spend:,.2f}",
        "recent_purchases": customer["recent_purchases"],
        "open_tickets": customer["open_tickets"],
        "open_ticket_count": open_ticket_count,
        "rep_notes": customer["notes"],
        "alert": "⚠️ Frustrated customer — escalate carefully" if open_ticket_count >= 2 else None
    }


def generate_interaction_brief(customer_id: str, interaction_type: str = "support") -> dict:
    """
    Generates a pre-interaction brief combining customer intelligence
    with relevant policy context for a rep about to contact a customer.

    Args:
        customer_id: The customer's ID number
        interaction_type: "support", "sales", or "retention"

    Returns:
        A structured brief with customer summary, recommended approach, and policy reminders
    """
    customer = CUSTOMER_DB.get(customer_id)

    if not customer:
        return {"found": False, "message": f"Customer {customer_id} not found."}

    open_tickets = customer["open_tickets"]
    tier = customer["tier"]
    ltv = customer["lifetime_value"]

    # Determine escalation path based on policies + customer tier
    if tier in ["Gold", "Platinum"]:
        escalation_note = "VIP customer — route directly to Tier 2 per escalation SOP."
    else:
        escalation_note = "Standard customer — begin at Tier 1."

    # Recommended approach by interaction type
    approaches = {
        "support": f"Lead with empathy. Acknowledge open issue(s) immediately. {escalation_note}",
        "sales": f"LTV ${ltv:,.2f} — strong upsell candidate. Reference recent purchase history.",
        "retention": f"At-risk signals detected. Prioritize resolution over policy. Offer goodwill gesture if needed."
    }

    return {
        "customer_name": customer["name"],
        "tier": tier,
        "lifetime_value": f"${ltv:,.2f}",
        "open_issues": len(open_tickets),
        "ticket_summary": [t["issue"] for t in open_tickets],
        "recommended_approach": approaches.get(interaction_type, approaches["support"]),
        "rep_notes": customer["notes"],
        "policy_reminder": escalation_note
    }


# ─────────────────────────────────────────────
# THREE HEADS OF CeREbrus
# ─────────────────────────────────────────────

knowledge_agent = LlmAgent(
    name="knowledge_base_agent",
    model="gemini-2.5-flash",
    description="Searches internal enterprise knowledge: policies, SOPs, vendor contracts, and compliance docs.",
    instruction="""You are CeREbrus Head 1 — the Knowledge Guardian.

    You answer employee questions by searching the internal knowledge base.

    When a question comes in:
    1. Use search_knowledge_base to find the relevant document
    2. Present the answer clearly — lead with the direct answer, then cite the source doc
    3. If policy has conditions or exceptions, call them out explicitly
    4. Always include the document name and last updated date so employees know they have current info
    5. If no document is found, say so clearly and suggest who to contact

    Tone: precise, professional, concise. You are the source of truth.""",
    tools=[search_knowledge_base],
)

customer_intel_agent = LlmAgent(
    name="customer_intel_agent",
    model="gemini-2.5-flash",
    description="Retrieves customer profiles including purchase history, support tickets, tier status, and rep notes.",
    instruction="""You are CeREbrus Head 2 — the Customer Intelligence Head.

    You give retail reps a full picture of any customer before or during an interaction.

    When given a customer ID:
    1. Use get_customer_profile to pull their full profile
    2. Lead with the most important context: tier, open tickets, any alerts
    3. Summarize recent purchases naturally — don't just list raw data
    4. Surface rep notes and any friction signals (multiple tickets, escalations)
    5. End with one clear recommended next step for the rep

    Tone: briefing-style, fast, scannable. A rep should be ready to engage in 30 seconds.""",
    tools=[get_customer_profile],
)

summary_agent = LlmAgent(
    name="summary_agent",
    model="gemini-2.5-flash",
    description="Generates pre-interaction briefs and cross-source intelligence summaries for retail reps.",
    instruction="""You are CeREbrus Head 3 — the Synthesis Head.

    You produce structured briefs that combine customer intelligence with relevant policy context.

    When asked for a brief or summary:
    1. Use generate_interaction_brief with the customer ID and interaction type
    2. Structure the output clearly: Customer Snapshot → Key Issues → Recommended Approach → Policy Reminders
    3. Flag any risk signals (frustrated customer, high LTV, VIP tier) at the top
    4. Keep it under 150 words — reps are reading this before a live call

    Tone: decisive, scannable, no fluff. Every word should help the rep.""",
    tools=[generate_interaction_brief],
)

# ─────────────────────────────────────────────
# ORCHESTRATOR — THE GATE
# ─────────────────────────────────────────────

root_agent = LlmAgent(
    name="cerebrus_orchestrator",
    model="gemini-2.5-flash",
    description="CeREbrus — Enterprise Retail Intelligence Platform. Three agents, one interface.",
    instruction="""You are CeREbrus — an enterprise retail intelligence platform built for retail operations teams.

    You have three specialist agents (three heads):
    1. knowledge_base_agent  — answers questions about internal policies, SOPs, vendor contracts, compliance
    2. customer_intel_agent  — retrieves customer profiles, purchase history, open tickets, and rep notes
    3. summary_agent         — generates pre-interaction briefs combining customer data + policy context

    How to route:
    - Policy, SOP, vendor contract, compliance question → knowledge_base_agent
    - "Tell me about customer [ID]", purchase history, open tickets → customer_intel_agent
    - "Brief me on customer [ID]", pre-call prep, interaction summary → summary_agent
    - Ambiguous query combining both policy + customer context → summary_agent

    Rules:
    - Always route to a specialist. Never answer policy or customer questions yourself.
    - If a customer ID is mentioned, extract it and pass it to the correct agent.
    - If the user seems unsure what to ask, present the three capabilities clearly.

    Opening: When greeted, introduce CeREbrus in one sentence and list the three things it can do.
    Tagline: "Three heads. One source of truth." """,
    sub_agents=[knowledge_agent, customer_intel_agent, summary_agent],
)
