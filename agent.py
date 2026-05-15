"""
CeREbrus — Enterprise Retail Intelligence Platform
lablab.ai Hackathon | Track 2: AI Agents with Google AI Studio
Powered by Google ADK + Gemini 3 (Pro for reasoning, Flash for retrieval)

"Cerberus in mythology had three heads guarding the gate.
 CeREbrus has three agents guarding your enterprise data."

Architecture:
  Orchestrator (CeREbrus)
  ├── Head 1: Knowledge Base Agent   — RAG over internal policy docs, SOPs, vendor contracts
  ├── Head 2: Customer Intel Agent   — purchase history, support tickets, behavior patterns
  └── Head 3: Summary Agent          — pre-interaction briefs, cross-source synthesis
"""

import os
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.adk.agents import LlmAgent

load_dotenv()

# Model selection — Gemini 3.x preview as of 2026-05.
# Pro for routing/synthesis (cross-source reasoning), Flash for retrieval roles.
# Swap in one place if a preview model misbehaves.
MODEL_REASONING = "gemini-3.1-pro-preview"
MODEL_FAST = "gemini-3-flash-preview"

EMBEDDING_MODEL = "gemini-embedding-001"
RAG_MIN_SIMILARITY = 0.55

_genai_client = None
_doc_embeddings_cache = None


def _genai():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _genai_client


def _embed(texts: list[str]) -> np.ndarray:
    result = _genai().models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])


def _doc_embeddings() -> tuple[list[str], np.ndarray]:
    """Lazy-init policy doc embeddings; cached for the lifetime of the process."""
    global _doc_embeddings_cache
    if _doc_embeddings_cache is None:
        keys = list(POLICY_DOCS.keys())
        texts = [
            f"{POLICY_DOCS[k]['title']}. {POLICY_DOCS[k]['content']}"
            for k in keys
        ]
        _doc_embeddings_cache = (keys, _embed(texts))
    return _doc_embeddings_cache

# ─────────────────────────────────────────────
# MOCK DATA LAYER
# Designed to reflect enterprise scale:
# multi-region customers, diverse tiers, cross-department policies
# In production: replace with real vector DB (Pinecone, Weaviate)
# and CRM/ERP API (Salesforce, SAP, HubSpot) integrations
# ─────────────────────────────────────────────

POLICY_DOCS = {

    # ── CUSTOMER-FACING POLICIES ──────────────────────────────────────────
    "return_policy": {
        "title": "Return & Refund Policy v2.3 — Enterprise Standard",
        "content": (
            "Standard return window is 30 days with receipt across all store locations and online orders. "
            "Electronics have a 15-day return window. Appliances have a 48-hour defect return window. "
            "Items purchased during promotional events (including Black Friday, Cyber Monday, clearance) "
            "are eligible for store credit only — no cash refunds. "
            "Opened software, digital downloads, gift cards, and personalized items are non-returnable. "
            "Damaged or defective items are eligible for full refund or replacement regardless of window. "
            "Manager override required for returns exceeding $500. District manager approval for $1,000+. "
            "Online returns ship free via prepaid label. In-store returns accepted at any location nationwide."
        ),
        "last_updated": "2025-11-01"
    },

    "loyalty_program": {
        "title": "Loyalty & Rewards Program — Tier Guidelines v3.1",
        "content": (
            "Four membership tiers: Standard (0–$999 annual spend), Silver ($1,000–$4,999), "
            "Gold ($5,000–$19,999), Platinum ($20,000+). "
            "Points accrual: 1 point per $1 Standard; 1.5x Silver; 2x Gold; 3x Platinum. "
            "Points expire after 18 months of account inactivity. "
            "Platinum members receive: dedicated support line, free expedited shipping, "
            "early access to sales (48 hours), annual $200 rewards credit, and assigned account manager. "
            "Gold members receive: free standard shipping, early access to sales (24 hours), "
            "and priority support queue. "
            "Tier downgrades occur annually on the account anniversary date if spend thresholds are not met. "
            "Corporate/business accounts qualify for separate contract pricing — contact B2B team."
        ),
        "last_updated": "2026-01-15"
    },

    "price_match": {
        "title": "Price Match & Competitive Pricing Policy v1.6",
        "content": (
            "Price match honored for identical items (same brand, model, SKU) from major national retailers "
            "including Amazon, Best Buy, Target, Walmart, and Costco. "
            "Price match must be requested at time of purchase or within 14 days of purchase. "
            "Excludes: marketplace third-party sellers, open-box/refurbished items, clearance pricing, "
            "membership-only prices, and limited flash sales. "
            "Online price match: submit via customer portal with competitor URL. "
            "In-store: rep verifies live pricing on store device before approving. "
            "Price match cap: $2,500 per transaction. Manager approval required above $500."
        ),
        "last_updated": "2025-09-10"
    },

    # ── OPERATIONS & VENDOR POLICIES ─────────────────────────────────────
    "vendor_terms": {
        "title": "Vendor Agreement — TechSupply Co. (Contract #VS-2024-089)",
        "content": (
            "Net-30 payment terms. Minimum order quantity: 500 units per SKU. "
            "Price lock guaranteed through Q3 2026. "
            "Penalty clause: 2% per week for late payments beyond 45 days. "
            "Renewal date: September 1, 2026. Exclusive distribution rights for Southeast region. "
            "SLA: 98% on-time delivery or vendor absorbs shipping cost differential. "
            "Product defect rate threshold: 1.5% — vendor liable for replacement costs above threshold. "
            "Annual renegotiation window: July 1–August 15. Volume discount tiers: "
            "5,000+ units/month = 8% discount; 10,000+ units/month = 14% discount."
        ),
        "last_updated": "2024-09-01"
    },

    "inventory_policy": {
        "title": "Inventory Management & Replenishment SOP v4.2",
        "content": (
            "Automatic reorder triggered when stock falls below 15% of monthly average sales velocity. "
            "Safety stock minimum: 7 days of average demand for top-100 SKUs; 3 days for all others. "
            "Markdown triggers: items with 90+ days on shelf automatically flagged for 20% discount review. "
            "Dead stock (180+ days): escalated to regional merchandising manager for clearance approval. "
            "Shrinkage threshold: stores exceeding 1.8% shrinkage rate trigger loss prevention audit. "
            "Seasonal inventory planning cycle begins 16 weeks prior to peak season. "
            "Cross-store transfers: approved by regional ops manager for quantities above 50 units. "
            "Out-of-stock customer notification: automated email sent within 2 hours of backorder status."
        ),
        "last_updated": "2026-02-28"
    },

    "shipping_fulfillment": {
        "title": "Shipping & Fulfillment Standards v2.9",
        "content": (
            "Standard shipping: 5–7 business days. Expedited: 2–3 business days. Overnight available on orders placed before 1PM EST. "
            "Free standard shipping on all orders over $49. "
            "Same-day delivery available in 42 metro markets via third-party courier partnership. "
            "Buy Online Pick Up In Store (BOPIS): ready within 2 hours for in-stock items. "
            "Large item delivery (appliances, furniture): scheduled delivery within 7–14 days; "
            "white-glove service available for $79 fee. "
            "International shipping: available to Canada, UK, and Australia. Duties are customer responsibility. "
            "Lost package threshold: carrier investigation initiated after 5 business days past expected delivery."
        ),
        "last_updated": "2025-12-01"
    },

    # ── COMPLIANCE & EMPLOYEE POLICIES ────────────────────────────────────
    "employee_escalation": {
        "title": "Customer Escalation SOP v1.8",
        "content": (
            "Tier 1: Front-line rep handles complaints and refunds under $200. Resolution target: 10 minutes. "
            "Tier 2: Supervisor approval required for refunds $200–$999 or second-contact complaints. "
            "Tier 3: District manager sign-off for refunds over $1,000 or repeat escalations (3+ contacts). "
            "Tier 4: VP Customer Experience for legal threats, media escalations, or corporate account issues. "
            "All escalations must be logged in CRM within 2 hours with root cause code. "
            "VIP customers (Gold/Platinum tier) skip Tier 1 — route directly to Tier 2. "
            "Response SLA by tier: Tier 1 same day; Tier 2 within 4 hours; Tier 3 within 24 hours. "
            "Goodwill gestures: reps authorized up to $50 store credit without approval; $51–$200 needs supervisor."
        ),
        "last_updated": "2025-06-15"
    },

    "compliance": {
        "title": "Data Privacy, PCI & Security Compliance Guidelines v2.1",
        "content": (
            "Never store full credit card numbers in CRM notes or emails. Last 4 digits only. "
            "PCI DSS Level 1 compliance required for all payment integrations — annual audit mandatory. "
            "Customer data requests (CCPA/GDPR): fulfilled within 30 days; deletion requests within 45 days. "
            "Screen recordings of customer interactions retained for 90 days then auto-deleted. "
            "Employee access to customer financial data restricted by role-based permissions. "
            "Data breach protocol: IT Security notified within 1 hour; customers notified within 72 hours per GDPR. "
            "Biometric data (where collected) stored separately with enhanced encryption (AES-256). "
            "Third-party vendor data sharing: requires signed DPA (Data Processing Agreement) on file."
        ),
        "last_updated": "2025-03-20"
    },
}

CUSTOMER_DB = {

    # ── PLATINUM TIER ─────────────────────────────────────────────────────
    "2034": {
        "name": "Sofia Reyes",
        "tier": "Platinum",
        "region": "West Coast",
        "lifetime_value": 38400.00,
        "account_since": "2018",
        "recent_purchases": [
            {"item": "MacBook Pro 16\"", "amount": 2499.00, "date": "2026-05-01", "status": "delivered"},
            {"item": "USB-C Hub Bundle", "amount": 89.99, "date": "2026-05-01", "status": "delivered"},
            {"item": "4K Monitor — Dell UltraSharp", "amount": 749.00, "date": "2026-03-14", "status": "delivered"},
            {"item": "Ergonomic Chair Pro", "amount": 899.00, "date": "2026-02-20", "status": "delivered"},
        ],
        "open_tickets": [],
        "notes": "Executive buyer for a mid-size tech firm. Bulk purchaser — typically orders 10–20 units. Contract pricing eligible. Prefers dedicated account manager contact. Fast decision-maker."
    },

    "1105": {
        "name": "Raymond Okafor",
        "tier": "Platinum",
        "region": "Northeast",
        "lifetime_value": 52100.00,
        "account_since": "2016",
        "recent_purchases": [
            {"item": "Samsung 85\" QLED TV", "amount": 3299.99, "date": "2026-04-30", "status": "delivered"},
            {"item": "Home Theater Receiver", "amount": 1199.00, "date": "2026-04-30", "status": "delivered"},
            {"item": "Smart Home Starter Kit", "amount": 499.00, "date": "2026-03-08", "status": "delivered"},
        ],
        "open_tickets": [
            {"id": "TK-8801", "issue": "TV delivery damaged corner of unit — requesting replacement", "status": "in-progress", "created": "2026-05-02"},
        ],
        "notes": "Highest LTV customer in Northeast region. Interior designer — buys for clients. Extremely detail-oriented. Has escalated twice historically but resolved positively both times."
    },

    # ── GOLD TIER ─────────────────────────────────────────────────────────
    "4821": {
        "name": "Marcus Johnson",
        "tier": "Gold",
        "region": "Southeast",
        "lifetime_value": 14820.50,
        "account_since": "2019",
        "recent_purchases": [
            {"item": "65-inch OLED TV", "amount": 1899.99, "date": "2026-04-10", "status": "delivered"},
            {"item": "Soundbar Pro X", "amount": 349.99, "date": "2026-03-22", "status": "delivered"},
            {"item": "Extended Warranty Pack", "amount": 199.00, "date": "2026-04-10", "status": "active"},
            {"item": "Smart Doorbell Camera", "amount": 229.99, "date": "2026-02-14", "status": "delivered"},
        ],
        "open_tickets": [
            {"id": "TK-9921", "issue": "TV remote not pairing after firmware update", "status": "open", "created": "2026-04-18"},
        ],
        "notes": "Loyal customer since 2019. Prefers email communication. Sensitive about wait times. Strong upsell candidate for smart home ecosystem products."
    },

    "3388": {
        "name": "Angela Torres",
        "tier": "Gold",
        "region": "Midwest",
        "lifetime_value": 9650.00,
        "account_since": "2021",
        "recent_purchases": [
            {"item": "French Door Refrigerator", "amount": 1549.00, "date": "2026-04-22", "status": "delivered"},
            {"item": "Dishwasher — Bosch 500 Series", "amount": 899.00, "date": "2026-04-22", "status": "delivered"},
            {"item": "5-Year Appliance Protection Plan", "amount": 299.00, "date": "2026-04-22", "status": "active"},
        ],
        "open_tickets": [
            {"id": "TK-9955", "issue": "Refrigerator ice maker not producing ice after 2 weeks", "status": "open", "created": "2026-05-06"},
        ],
        "notes": "Recently renovated home — high appliance spend this quarter. Active warranty holder. Likely to purchase washer/dryer next. Respond promptly — appliance issues are high urgency."
    },

    # ── SILVER TIER ───────────────────────────────────────────────────────
    "5590": {
        "name": "James Whitfield",
        "tier": "Silver",
        "region": "South",
        "lifetime_value": 3200.00,
        "account_since": "2022",
        "recent_purchases": [
            {"item": "Dyson V15 Vacuum", "amount": 749.99, "date": "2026-05-03", "status": "in-transit"},
            {"item": "Air Purifier — Levoit 400S", "amount": 219.99, "date": "2026-04-10", "status": "delivered"},
        ],
        "open_tickets": [
            {"id": "TK-9970", "issue": "Vacuum not yet delivered — 4 days past expected date", "status": "open", "created": "2026-05-08"},
        ],
        "notes": "Growing account — spend increased 60% YoY. Proactive communicator. Late delivery situation needs immediate attention to preserve loyalty trajectory."
    },

    # ── STANDARD TIER ─────────────────────────────────────────────────────
    "7710": {
        "name": "Derek Park",
        "tier": "Standard",
        "region": "West",
        "lifetime_value": 1240.00,
        "account_since": "2025",
        "recent_purchases": [
            {"item": "Gaming Headset Z", "amount": 129.99, "date": "2026-04-28", "status": "delivered"},
            {"item": "Controller Charging Dock", "amount": 39.99, "date": "2026-03-10", "status": "delivered"},
        ],
        "open_tickets": [
            {"id": "TK-9988", "issue": "Headset defective — no audio in left ear", "status": "escalated", "created": "2026-05-02"},
            {"id": "TK-9901", "issue": "Return request denied — outside window", "status": "closed", "created": "2026-04-15"},
        ],
        "notes": "Two escalations in 30 days. Frustrated customer — handle with care. Consider goodwill gesture to prevent churn."
    },

    "8842": {
        "name": "Priya Nair",
        "tier": "Standard",
        "region": "Northeast",
        "lifetime_value": 540.00,
        "account_since": "2026",
        "recent_purchases": [
            {"item": "Instant Pot Duo 7-in-1", "amount": 89.99, "date": "2026-05-10", "status": "delivered"},
            {"item": "Kitchen Scale", "amount": 24.99, "date": "2026-05-10", "status": "delivered"},
        ],
        "open_tickets": [],
        "notes": "New customer — first purchase May 2026. No interaction history. Strong onboarding opportunity. Purchased cookware category — candidate for kitchen appliance cross-sell."
    },
}

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def search_knowledge_base(query: str) -> dict:
    """
    Semantic search across the enterprise knowledge base using
    Gemini embeddings (gemini-embedding-001) + cosine similarity.

    Doc embeddings are computed once on first call and cached for the
    process lifetime; per-query cost is one embedding API call.

    Args:
        query: Natural language question from an employee

    Returns:
        Matching document with similarity score, or a no-match response
        if no doc clears the similarity threshold.
    """
    keys, doc_emb = _doc_embeddings()
    q_emb = _embed([query])[0]

    sims = doc_emb @ q_emb / (np.linalg.norm(doc_emb, axis=1) * np.linalg.norm(q_emb))
    top_idx = int(np.argmax(sims))
    top_sim = float(sims[top_idx])

    if top_sim < RAG_MIN_SIMILARITY:
        return {
            "found": False,
            "message": "No matching policy found (semantic similarity below threshold). Try rephrasing or contact your ops manager.",
            "best_match_score": round(top_sim, 3),
            "suggestion": (
                "Available topics: returns & refunds, loyalty program, price matching, "
                "vendor contracts, inventory management, shipping & fulfillment, "
                "escalation procedures, data & compliance"
            )
        }

    doc = POLICY_DOCS[keys[top_idx]]
    return {
        "found": True,
        "document": doc["title"],
        "content": doc["content"],
        "last_updated": doc["last_updated"],
        "source": "CeREbrus Internal Knowledge Base (semantic search)",
        "match_score": round(top_sim, 3),
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


def analyze_customer_portfolio(focus: str = "all") -> dict:
    """
    Returns the entire customer portfolio with computed signals so the
    synthesis agent can reason across all customers in one pass —
    identifying churn risk, upsell candidates, accounts needing immediate
    attention, or any other cross-cutting pattern.

    Use this tool ONLY for portfolio-wide questions like:
      "Who's at churn risk?"
      "Top 3 upsell candidates?"
      "Which accounts need immediate attention?"
      "Show me high-value customers with unresolved issues"

    Do NOT use this for single-customer questions — those go to
    get_customer_profile or generate_interaction_brief.

    Args:
        focus: Optional analytical lens. One of:
            "churn_risk" - emphasize at-risk signals
            "upsell"     - emphasize buying-pattern signals
            "retention"  - emphasize relationship signals
            "all"        - balanced view (default)

    Returns:
        Portfolio-level metrics plus enriched per-customer signals
        (LTV, recent spend, ticket age, escalation count, days since
        last purchase) for the agent to reason over.
    """
    today = datetime.now().date()

    portfolio = []
    for cid, c in CUSTOMER_DB.items():
        open_tickets = c["open_tickets"]
        escalated = sum(1 for t in open_tickets if t.get("status") == "escalated")
        in_progress = sum(1 for t in open_tickets if t.get("status") == "in-progress")
        recent_spend = sum(p["amount"] for p in c["recent_purchases"])

        if c["recent_purchases"]:
            most_recent_str = max(p["date"] for p in c["recent_purchases"])
            days_since_purchase = (today - datetime.fromisoformat(most_recent_str).date()).days
        else:
            days_since_purchase = None

        if open_tickets:
            oldest_str = min(t["created"] for t in open_tickets)
            oldest_ticket_age_days = (today - datetime.fromisoformat(oldest_str).date()).days
        else:
            oldest_ticket_age_days = 0

        portfolio.append({
            "customer_id": cid,
            "name": c["name"],
            "tier": c["tier"],
            "region": c.get("region", "—"),
            "lifetime_value": c["lifetime_value"],
            "recent_spend_90d": round(recent_spend, 2),
            "open_ticket_count": len(open_tickets),
            "escalated_tickets": escalated,
            "in_progress_tickets": in_progress,
            "ticket_topics": [t["issue"] for t in open_tickets],
            "oldest_open_ticket_days": oldest_ticket_age_days,
            "days_since_last_purchase": days_since_purchase,
            "account_since": c.get("account_since", "—"),
            "rep_notes": c["notes"],
        })

    portfolio.sort(key=lambda x: x["lifetime_value"], reverse=True)

    return {
        "as_of": today.isoformat(),
        "focus": focus,
        "total_customers": len(portfolio),
        "total_lifetime_value": round(sum(p["lifetime_value"] for p in portfolio), 2),
        "customers_with_open_tickets": sum(1 for p in portfolio if p["open_ticket_count"] > 0),
        "customers_with_escalations": sum(1 for p in portfolio if p["escalated_tickets"] > 0),
        "portfolio": portfolio,
    }


# ─────────────────────────────────────────────
# THREE HEADS OF CeREbrus
# ─────────────────────────────────────────────

knowledge_agent = LlmAgent(
    name="knowledge_base_agent",
    model=MODEL_FAST,
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
    model=MODEL_FAST,
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
    model=MODEL_REASONING,
    description="Generates single-customer briefs AND portfolio-wide synthesis (churn risk, upsell candidates, accounts needing attention).",
    instruction="""You are CeREbrus Head 3 — the Synthesis Head.

    You handle two distinct kinds of questions:

    A) SINGLE-CUSTOMER BRIEFS — questions like "Brief me on customer 4821",
       "Pre-call prep for 7710":
       1. Use generate_interaction_brief with the customer ID and interaction type
       2. Structure: Customer Snapshot → Key Issues → Recommended Approach → Policy Reminders
       3. Flag risk signals (frustrated, high LTV, VIP tier) at the top
       4. Keep under 150 words

    B) PORTFOLIO-WIDE SYNTHESIS — questions like "Who's at churn risk?",
       "Top upsell candidates?", "Which accounts need attention this week?",
       "High-value customers with unresolved issues?":
       1. Use analyze_customer_portfolio with the appropriate focus
       2. Reason across the returned signals — DO NOT just list everyone
       3. Identify the customers that match the question and rank them
       4. For each ranked customer, cite the specific signals justifying inclusion
          (e.g. "LTV $52,100, ticket open 12 days, in-progress status")
       5. Output format: Top 3 with reasoning → optional 1-line tail for the rest
       6. End with one concrete next action a manager can take today

    Tone: decisive, scannable, no fluff. Every word should help the rep or manager.""",
    tools=[generate_interaction_brief, analyze_customer_portfolio],
)

# ─────────────────────────────────────────────
# ORCHESTRATOR — THE GATE
# ─────────────────────────────────────────────

def write_portfolio_snapshot(path: str = None) -> str:
    """Write the portfolio data to JSON for the dashboard UI to load on init."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "portfolio_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analyze_customer_portfolio(), f, indent=2, default=str)
    return path


# Also runs as a side-effect of import, so ADK web regenerates snapshot
# whenever it (lazy) imports the agent — typically on first chat request.
try:
    write_portfolio_snapshot()
except Exception:
    pass


if __name__ == "__main__":
    p = write_portfolio_snapshot()
    print(f"Wrote portfolio snapshot: {p}")


root_agent = LlmAgent(
    name="cerebrus_orchestrator",
    model=MODEL_REASONING,
    description="CeREbrus — Enterprise Retail Intelligence Platform. Three agents, one interface.",
    instruction="""You are CeREbrus — an enterprise retail intelligence platform built for retail operations teams.

    You have three specialist agents (three heads):
    1. knowledge_base_agent  — answers questions about internal policies, SOPs, vendor contracts, compliance
    2. customer_intel_agent  — retrieves single-customer profiles (purchase history, tickets, notes)
    3. summary_agent         — generates single-customer briefs AND portfolio-wide synthesis
                                (churn risk, top accounts, who needs attention)

    How to route:
    - Policy, SOP, vendor contract, compliance question → knowledge_base_agent
    - "Tell me about customer [ID]", single-customer profile lookup → customer_intel_agent
    - "Brief me on customer [ID]", pre-call prep → summary_agent
    - PORTFOLIO/CROSS-CUSTOMER questions ("who's at churn risk", "top upsell
      candidates", "which accounts need attention", "high-value customers with
      open issues", anything spanning multiple customers) → summary_agent
    - Ambiguous query combining policy + customer context → summary_agent

    Rules:
    - Always route to a specialist. Never answer policy or customer questions yourself.
    - If a customer ID is mentioned, extract it and pass it to the correct agent.
    - If the user seems unsure what to ask, present the four capabilities clearly:
      knowledge lookups, single-customer profile, single-customer brief, portfolio synthesis.

    Opening: When greeted, introduce CeREbrus in one sentence and list what it can do.
    Tagline: "Three heads. One source of truth." """,
    sub_agents=[knowledge_agent, customer_intel_agent, summary_agent],
)
