# Original project requirements

This is a readable transcription of the original assignment supplied by the user
on 2026-10-29. Wording is lightly normalized; required tasks, examples, optional
items, and outcomes retain their original meaning. It is the fixed assignment
baseline, **not** a record of implementation decisions or added features.

Source screenshots (original files currently supplied from outside this repository):

| Original filename | Content |
| --- | --- |
| `Screenshot 2026-10-29 160742.png` | Technical specification and general requirements |
| `Screenshot 2026-10-29 160800.png` | Stage 1 |
| `Screenshot 2026-10-29 160812.png` | Stage 2 |
| `Screenshot 2026-10-29 160820.png` | Stage 3 |
| `Screenshot 2026-10-29 160834.png` | Stage 4 |

The screenshots are the source of truth if this transcription is disputed. Preserve
original images unchanged if archived with the project. Record user-approved additions
and choices separately in AGENTS.md; do not rewrite the original assignment to match code.
Assignment dates shown: October 24–November 7, 2026 (14 days). Screenshot UI status
labels are not evidence of the repository's implementation status.

## Goal and general requirements

Develop an intelligent chatbot that interacts with users, provides parking-space
information, handles reservations, and involves a human administrator for confirmation
(human-in-the-loop). The project has four stages.

1. **Programming language:** Python.
2. **Frameworks:** LangChain and LangGraph.
3. **Architecture:** Retrieval-Augmented Generation (RAG).
4. **Vector database:** recommended options include Milvus, Pinecone, or Weaviate.
5. **General features:**
   - Provide general parking information, working hours, prices, space availability,
     and location.
   - Interactively collect name, surname, car number, and reservation period.
   - Prevent exposure of sensitive data, including private information stored in the
     vector database.
   - Evaluate system performance, for example request latency and retrieval accuracy.
6. **Submission and extra credit:** see the shared section below.
7. Poor-quality code, code too basic to be practical, or critical errors may reduce the grade.

## Stage 1: Creation of a RAG system and chatbot

### Tasks

1. Implement the basic RAG chatbot architecture for interacting with users.
2. Integrate a vector database for storing information.
   - **Optional improvement:** split source data into static and dynamic types.
     Static examples: general information, parking details, location, booking process;
     store these in the vector database. Dynamic examples: space availability,
     working hours, prices; store these in a SQL database.
3. Provide information and collect user inputs for reservations interactively.
4. Implement guardrails and filtering to prevent sensitive-data exposure, for example
   using pretrained NLP models for text analysis.
5. Evaluate the RAG system: performance testing and response-accuracy measurement,
   for example Recall@K and Precision.

### Outcome

- A working chatbot that provides basic information and interacts with users.
- Data-protection functionality.
- An evaluation report on system performance.

## Stage 2: Implementation of a human-in-the-loop agent

### Tasks and key features

1. Create a second agent using LangChain to interact with the administrator.
2. Send a reservation request to the administrator and receive confirmation or refusal,
   for example via an email server, messenger, or REST API.
3. Integrate it with the first agent so requests are escalated to the human administrator
   after collecting the user's details.
4. Generate/send reservation confirmation requests, receive administrator responses,
   and maintain communication between the first and second agents.

### Outcome

- An automated system that contacts an administrator for reservation approval.

## Stage 3: Process confirmed reservations using an MCP server

### Tasks

1. Use an open-source MCP server that can write data to a file. Alternatively,
   develop a simple MCP server using Python + FastAPI to process confirmed reservations.
   - If an MCP server is not possible to implement, use tool/function calling to write
     data to a file.
2. Once the administrator approves the reservation through the second-agent workflow,
   write reservation details to a text file.
   - **File-entry format:** `Name | Car Number | Reservation Period | Approval Time`.
3. Ensure the server is secure, resistant to unauthorized access, and reliable.

### Outcome

- A functional MCP server integrated with the preceding agents, processing reservation
  data and saving it in storage (or the explicitly permitted file-writing fallback).

## Stage 4: Orchestrating all components via LangGraph

### Tasks

1. Implement orchestration of all components using LangGraph.
2. Fully integrate all stages: the RAG chatbot interacts with users; the second
   human-in-the-loop agent escalates requests to the administrator; MCP processes
   data after confirmation.
3. Implement the complete workflow logic. Example nodes: user interaction (RAG/chatbot
   context), administrator approval, and data recording.
4. Test the entire system workflow.

### Outcome

- A unified system whose components work together, with stable pipeline operation.

### System testing and documentation

- Load-test each component: interactive chatbot dialogue, administrator confirmation,
  and MCP recording/storage.
- Integration-test all steps during orchestration.
- Document usage, architecture, agent/server logic, setup, and deployment guidelines.

## Submission and extra-credit artifacts (all stages)

- For each task/stage, provide a GitHub or EPAM GitLab repository link in the answer field.
- **Extra credit:**
  - A PowerPoint presentation explaining how the solution works, with relevant screenshots.
  - A README with clear setup, usage, structure, and other project documentation.
  - Automated pytest or unittest tests, at least two tests per module.
  - CI/CD automation and/or infrastructure as code, for example Terraform.
- Poor-quality, impractically basic code or critical errors may reduce the grade.
