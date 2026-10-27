# parking-space-reservation-agent
The goal of the project is to develop an intelligent chatbot that can interact with users, provide information about parking spaces, handle the reservation process, and involve a human administrator for confirmation.

## System prompt

The user-facing prompt is defined in `commons/constants.py`. The approved architecture uses
two agents coordinated by LangGraph/Python code. Allocation, approval enforcement, reservation
state, and file formatting/writing are programmatic responsibilities, not LLM decisions.
The current prototype does not yet implement those services.

## Offline checks

With the project dependencies installed in `.venv`, run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These checks cover the prompt contract and system-message integration with a mocked model.
They do not make API calls or evaluate live model behavior, retrieval accuracy, or security enforcement.
