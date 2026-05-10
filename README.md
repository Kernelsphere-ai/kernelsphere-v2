# Kernelsphere v2

A Python framework for browser automation built around a different approach: read the page fresh at every step, reason through what's on screen, and act.

Instead of fixed selectors and predefined scripts, Kernelsphere builds a live graph of visible elements at each step, scores them semantically, and executes the right action. No assumptions about layout.

Website: [kernelsphere.ai](https://kernelsphere.ai/)

## How it works

At every step the agent:

1. Extracts all interactive elements from the live page including cross-frame and shadow DOM
2. Builds a page graph with confidence levels and visibility scores per element
3. Scores candidates semantically using sentence transformers
4. Uses Gemini to reason through ambiguous cases
5. Executes the action and verifies the state changed
6. Remembers what worked so it does not rediscover the same elements on the same site next time

## Stack

- **Playwright** - browser control
- **Gemini** - reasoning and content extraction
- **Sentence Transformers (BGE / MiniLM)** - semantic element scoring
- **TF-IDF + synonym expansion** - fallback scorer when neural model is unavailable
- **Selector memory** - persistent cache of successful selectors per host

## Installation

```bash
pip install playwright sentence-transformers
playwright install chromium
```

Clone the repo:

```bash
git clone https://github.com/Kernelsphere-ai/kernelsphere-v2.git
cd kernelsphere-v2
```

Add your Gemini API key to the `.env` file.

## Usage

### Step-based

```python
from browser_automation_agent import BrowserAutomationAgent, AutomationStep
from logging_config import setup_logging

setup_logging(level="INFO")

steps = [
    AutomationStep(action="type", intent="type email", value="user@example.com"),
    AutomationStep(action="type", intent="type password", value="secret"),
    AutomationStep(action="click", intent="click sign in"),
]

with BrowserAutomationAgent(headless=True) as agent:
    result = agent.run_task(url="https://example.com/login", steps=steps)
    print(f"Success: {result.success}")
    for step in result.steps:
        print(f"  {step.step.action}: ok={step.ok} changed={step.changed}")
```

### Goal-based

Pass a natural language goal and let the planner figure out the steps:

```python
with BrowserAutomationAgent(headless=True) as agent:
    execution = agent.run_user_goal(
        url="https://example.com",
        goal="Find the cheapest one-way flight from Berlin to Rome in June"
    )
    print(execution.extracted_answer)
```

## Running tasks from a dataset

```bash
python runner.py \
  --dataset tasks.jsonl \
  --output ./results \
  --log-level INFO
```

Task format (`.jsonl` or `.json`):

```json
{
  "task_id": "task-001",
  "url": "https://example.com",
  "goal": "Find the contact email on the about page",
  "steps": [
    { "action": "click", "intent": "click about page link" },
    { "action": "extract", "intent": "find the contact email address" }
  ]
}
```

CLI flags:

```
--dataset         Path to .jsonl or .json task file
--output          Directory for traces and results summary
--headed          Run with a visible browser window
--limit N         Run only the first N tasks
--log-level       DEBUG | INFO | WARNING | ERROR
--log-file        Optional path to write logs to a file
```

## Supported actions

| Action | Description |
|---|---|
| `click` | Click an element |
| `type` | Type a value into an input |
| `select` | Select an option from a dropdown |
| `check` | Check a checkbox |
| `uncheck` | Uncheck a checkbox |
| `scroll` | Scroll the page (`up`, `down`, `top`, `bottom`, or pixel amount) |
| `navigate` | Navigate to a URL |
| `extract` | Read page content and extract an answer using Gemini |


## License

MIT see [LICENSE](LICENSE)

## Contributing
Kernel Sphere is open source and maintained in public. Contributions are welcome across SDK features, docs, examples, and bug reports.

## Contact
For questions, feedback, or collaboration: lakshmiprasanna@kernelsphere.ai
