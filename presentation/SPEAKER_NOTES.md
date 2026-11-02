# Speaker notes — Parking Space Reservation Agent

Author: Narek Hovhannisyan. English, 14 slides, suggested total: 10 minutes.

Notes are also embedded in the PowerPoint. Timings are suggestions, not automatic slide advances.

Screenshots are user-approved presentation assets. Crops preserve the originals; the format diagram is a template, not output-file evidence.


## 01. Parking Space Reservation Agent

**Suggested time:** 25s (0:00–0:25)

This project connects a parking-information chatbot to a human approval workflow and approved-only file recording. It is a Python learning assignment built around LangChain, LangGraph, RAG and MCP, rather than a production parking business. I will explain the responsibilities of each component, show the Studio workflow, and finish with the measured results and their limits. The central distinction throughout is that an LLM can help with language, but it cannot approve or allocate a reservation. Those transitions are explicit and controlled.

**Sources:** PROJECT_REQUIREMENTS.md; README.md


## 02. Four stages, one complete workflow

**Suggested time:** 35s (0:25–1:00)

The original assignment has four stages. First, build a RAG chatbot that answers parking questions, collects the required details, filters sensitive data and has an evaluation report. Second, create a separate LangChain agent that communicates with the administrator. Third, record only human-approved reservations using MCP and the specified four-field text format. Fourth, integrate and test the workflow through LangGraph, including dialogue, review and recording load. Chroma, Studio review and a private stdio MCP server are the accepted implementation choices. Examples such as email or FastAPI were not additional mandatory features.

**Sources:** PROJECT_REQUIREMENTS.md, stages 1–4; AGENTS.md, accepted choices


## 03. Two agents. Deterministic coordination.

**Suggested time:** 50s (1:00–1:50)

There are two language-model agents, not a third orchestrator model. The parking agent handles user interaction and clarification. The administrator agent prepares a bounded review message from the canonical submitted request. Deterministic Python and LangGraph route the conversation, retrieve evidence, validate fields, create holds, apply human decisions and invoke recording. Chroma supplies public descriptions and hourly-price metadata. SQLite holds public dynamic facts and a separate private operational ledger. The recorder independently reads that ledger in read-only mode. Resources belong to the agent instance; conversation and request state belong to individual graph threads.

**Sources:** README.md, project structure/current workflow; AGENTS.md, agent ownership


## 04. RAG uses public facts—not reservation data

**Suggested time:** 40s (1:50–2:30)

The public corpus contains twenty fictional parking spaces with differentiated route descriptions, benefits and trade-offs. Their fixed USD hourly prices are numeric Chroma metadata, not embedded prose. SQLite provides hours and availability; missing hours remain unknown. Its private reservation table is never public RAG evidence. Python derives a bounded public-topic query rather than embedding names, vehicle identifiers or raw chat. It filters eligibility before retrieving up to three similarity-ranked options and guards the evidence. The existing extraction call distinguishes parking from unrelated conversation, so unrelated turns skip retrieval rather than receiving stale parking context.

**Sources:** README.md, persistent parking information/information retrieval; PARKING_DATA_MODEL.md


## 05. Filter first. Rank eligible options second.

**Suggested time:** 40s (2:30–3:10)

Availability and price compliance are code responsibilities. Python excludes publicly blocked spaces and conflicting holds, then combines those eligible IDs with an optional numeric hourly-price filter in Chroma before native top-three ranking. Under eight dollars is strict; up to eight dollars includes eight; supported ranges include both ends. Total-stay costs, ambiguous units and unsupported currency cues require clarification rather than unrestricted offers. The budget is optional search state, not a fifth required reservation field. Descriptive preferences such as carrying bags affect semantic ranking, not guaranteed accessibility or feature suitability. Python displays every selected ID, description and rate.

**Sources:** README.md, optional hourly price constraints; RAG_EVALUATION.md, budgets/benefits


## 06. Collect → validate → ask the user to confirm

**Suggested time:** 45s (3:10–3:55)

The four required fields are first name, surname, a user-provided vehicle identifier and duration in hours. The vehicle does not need a conventional plate format. This implementation deliberately uses immediate duration-only requests: it never asks the user for a date, start time or timezone. Model extraction proposes grounded values; Python validates them and owns the revisioned confirmation summary shown here. Explicit corrections invalidate confirmation, and a vague disagreement blocks submission until clarified. Saying yes confirms only the current details—it is neither a submission nor human administrator approval. Names and vehicle details shown in these user-approved screenshots belong to the demonstration.

**Sources:** README.md, reservation draft collection; presentation/confirming-user-info-question.png


## 07. The human administrator owns the decision

**Suggested time:** 45s (3:55–4:40)

After confirmation, the user selects a currently offered space with slash-submit and its ID. Python captures the server UTC time inside an atomic SQLite hold transaction and calculates the end from the confirmed duration. The administrator agent then prepares the review wording, while Python supplies the exact canonical facts through a typed LangGraph interrupt. In the same Studio thread, the operator selects approve or refuse and clicks Resume. No request ID needs to be copied and there is no default decision. Waiting for review consumes the requested hours; approval at or after the fixed end becomes elapsed, not approved.

**Sources:** README.md, Stage 2 submission/review; presentation/interrupting-for-admin-review.png


## 08. Only approval opens the recording gate

**Suggested time:** 45s (4:40–5:25)

Python invokes exactly one narrow MCP tool, record_approved_reservation, using internal request and thread identifiers. No language model chooses an arbitrary write tool, filename, body or path. The private stdio server independently reads SQLite in read-only mode and verifies a real human-approved canonical request. It owns the required four-field UTF-8 format: full name, vehicle identifier, original UTC reservation period and original UTC approval instant. The format shown here is a template, not a screenshot of an output file. Publication is atomic and does not overwrite existing files; matching-content replay provides idempotent file effects. Only a validated success receipt permits a recording-success reply.

**Sources:** README.md, Stage 3 approved recording; PROJECT_REQUIREMENTS.md, file-entry format


## 09. LangGraph makes each transition explicit

**Suggested time:** 45s (5:25–6:10)

This is the actual Studio graph, including the intentionally retained spelling parking_assitant in code. Information turns retrieve evidence before answering. Collection, confirmation and submitted outcomes have code-controlled branches. After a successful submission, the graph emits the pending response, prepares administrator communication and pauses for review. Approval routes to recording; refusal or an elapsed request bypasses it. Invalid review values create a fresh interrupt without changing the ledger. Review preparation is checkpointed separately, so resuming does not create another hold or regenerate communication. Slash-new starts a separate one-space request only after a decision, preserving sequential workflow semantics.

**Sources:** README.md, current graph/hold and decision rules; presentation/agent-diagram.png


## 10. Demo: browse, confirm, keep useful choices

**Suggested time:** 55s (6:10–7:05)

The user starts by asking for parking and mentioning heavy bags. Early browsing is allowed, and the chatbot displays options while collecting the missing details. The user then provides the names, vehicle identifier and forty-eight-hour duration, and confirms the summary. Python automatically rechecks the earlier choices and keeps eligible IDs in their original order; the user does not need to issue a mandatory second slash-options search. This screenshot preserves OUT-V01, G-A02 and OUT-P01. The user explicitly submits OUT-V01. The heavy-bags example also illustrates a ranking limit: OUT-V01's own description warns that it is less convenient for bags. Retrieval is not perfect feature matching.

**Sources:** README.md, option continuity; presentation/inital-question.png; presentation/confirming-user-info-answer.png


## 11. Demo: approval and refusal stay distinct

**Suggested time:** 45s (7:05–7:50)

These screenshots show the two administrator outcomes. Approval retains the allocation and original UTC period, then the MCP step returns a validated recording receipt and the chatbot reports recorded successfully. The refusal branch releases its pending hold and explicitly says that nothing was recorded. Human approval and file-writing success remain separate results: if recording fails, the approved allocation is retained and the reply must report failure or uncertainty, never pretend success. The selected policy allows one initial attempt and at most one transient retry, then a manual slash-retry-recording for the current approved request. There are no ordinary-chat or background retries.

**Sources:** presentation/admin-approved.png; presentation/admin-refuse.png; README.md, failures/retries


## 12. Practical safeguards, explicit trust limits

**Suggested time:** 55s (7:50–8:45)

Local Presidio, a pinned spaCy English model and deterministic rules guard public ingestion, selected evidence, model-copy history and generated replies. Necessary reservation fields can still reach the configured provider. Raw model tokens are suppressed from the guarded messages stream, but Studio, checkpoints and debug APIs remain trusted development surfaces. The recording service uses owner-only local permissions, approval checks, atomic publication and bounded subprocess sessions. The small synthetic privacy evaluation fully protected twelve of thirteen positive cases and falsely flagged none of eight benign cases. An unknown unlabeled vehicle identifier was missed. These are practical boundaries—not complete confidentiality, a sandbox or a separate authenticated administrator application.

**Sources:** GUARDRAIL_EVALUATION.md; README.md, data protection/recording safety


## 13. Measured evidence—not production claims

**Suggested time:** 45s (8:45–9:30)

The priced twenty-query retrieval evaluation measured Recall-at-three of 0.547 and Precision-at-three of 0.500. Two live-answer runs passed twelve of thirteen conversations, and all twelve deterministic budget cases passed. The latest live-answer run's whole-turn median was 5.574 seconds (14 turns), not a latency guarantee. Results are mixed: exact-ID lookup and two challenge paraphrases remain weak spots. Separately, the historical full-workflow local load completed ninety conversations across concurrency one, five and ten, producing forty-five approved files. Those load models and review responses were injected or scripted, not provider or human throughput. Current HTTP/stdio regression covers thirteen scenarios; the latest documented offline suite has 391 passes and nine opt-in skips. Real-provider approval/refusal smoke is separate, small-sample evidence.

**Sources:** RAG_EVALUATION.md, 2026-11-02; STAGE4_VALIDATION.md, historical load/current regression


## 14. Four stages. One explicit approval flow.

**Suggested time:** 30s (9:30–10:00)

The delivered assignment slice combines RAG parking assistance, a second LangChain communication agent, human review, approved-only MCP recording and LangGraph orchestration. The takeaway is separation of responsibilities: models assist with language, the human decides, and Python enforces state transitions and writes. The repository contains setup instructions, architecture and evaluation reports. Run the locked environment locally, configure credentials, connect the required VPN and launch LangGraph Studio. Chroma persists, but SQLite and generated recording output deliberately reset on application startup. The repository link comes from project documentation; this presentation does not verify publication of the current local changes. Thank you—questions are welcome.

**Sources:** README.md, setup/deployment/repository links; PROJECT_REQUIREMENTS.md
