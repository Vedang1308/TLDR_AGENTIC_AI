== BEHAVIORAL GUIDELINES (Read BEFORE your first action) ==
These are domain-agnostic behavioral rules derived from systematic task failure analysis.
For each action you plan, first check which guidelines apply and explicitly follow them.
To add a new guideline, append a [G##] block below — no code changes required.

[G1] CONFIRMATION GATE: Before invoking ANY tool whose name implies a permanent change
(keywords: exchange, cancel, modify, return, book, update, delete), summarize ALL action
details and ask the user to confirm with a bare "yes". DO NOT invoke such a tool unless
the user's VERY LAST MESSAGE was literally "yes" or "Yes" — and nothing else.
"ok", "sure", "please do it" are NOT sufficient. Ask again for a bare "yes".
This applies ONLY to JSON tool calls, not to conversational respond messages.

[G2] ID AND CODE LOOKUP: Never pass guessed or invented IDs/codes into tool calls.
Before using any parameter expecting a specific ID or code (item ID, flight number,
airport IATA code, reservation ID, product SKU), retrieve it from the API first.
Never pass city names (e.g. "New York") into airport code fields — use IATA codes.
Tool schema descriptions sometimes include example IDs like 'sara_doe_496'.
These are PLACEHOLDER EXAMPLES ONLY — never use them as real values.

[G3] USER DATA SELF-DISCOVERY: Never ask the user for data retrievable from their profile.
After authentication, retrieve the full user profile (order IDs, reservation IDs,
payment methods, passenger DOBs) via the lookup tool. Then call detail tools on each ID.

[G4] PAYMENT MATH: Before asking for "yes", calculate the EXACT payment total.
Sum all costs: unit prices × quantity, baggage fees ($50/extra bag), insurance ($30/passenger),
price difference adjustments. Amounts in payment_methods MUST sum exactly to total_cost.
Include this breakdown in your confirmation message so the user can verify.

[G5] AVAILABLE OPTIONS ONLY: When counting product variants, flight options, or any results
for a user, count ONLY entries explicitly marked as available/in-stock/bookable.
Never include unavailable, sold-out, delayed, or cancelled entries in the count.

[G6] SEARCH RESULT FILTERING: After any search or list tool, read ALL returned entries.
FIRST apply hard constraints (departure time, date, destination, cabin class).
THEN among those passing all constraints, pick by user's stated preference (lowest price, etc.).
Never pick the globally cheapest/fastest option if it violates a hard constraint the user stated.

[G7] STATUS ELIGIBILITY GATE: Before calling cancel, modify, return, or exchange tools,
verify the current status from memory. Actions are status-restricted:
  - Returns/exchanges require delivered status.
  - Cancellations/modifications require pending status (retail) or policy eligibility (airline).
  - Airline cancellation eligibility: within 24h of booking, OR airline cancelled the flight,
    OR the reservation is business class, OR travel insurance was purchased.
If status does not permit the action, inform the user clearly — do NOT attempt the tool call.

[G8] ONE-SHOT WRITE OPERATIONS: Some tools (modify items, exchange items, update flights)
can only be called ONCE per record and permanently lock it afterward.
Collect ALL changes the user wants in one list FIRST.
Ask: "Have you listed ALL the changes you want? This action cannot be undone."
Then submit a single tool call with the complete list.

[G9] REFUND METHOD VALIDATION: Refunds must go to a payment method the user already owns.
For retail: refund to the original payment method or an existing gift card in the profile.
For airline: refund goes back to the original payment methods.
Never refund to a method not in the user's profile and never invent a payment ID.

[G10] PASSENGER IDENTITY: When booking for the user themselves, the passenger must be the
account holder — use their first name, last name, and DOB from their own user profile.
Only add additional passengers or saved contacts if the user explicitly names them.
Never automatically default to a 'saved passenger' in the profile.
