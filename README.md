# Screenshot to Todoist Task

Automatically extract actionable tasks from screenshots and create them in Todoist using GPT-4 Vision.

## Overview

This FastAPI service accepts screenshot uploads, uses OpenAI's vision API to detect actionable tasks, and automatically creates them in Todoist using natural language parsing. Perfect for quickly capturing tasks from Slack messages, emails, Jira tickets, or any on-screen content.

## Features

- 🖼️ **Vision-powered task extraction** – GPT-4o analyzes screenshots to identify action items
- ✅ **Structured output** – Schema-validated responses ensure reliability
- 🎯 **Smart filtering** – Only creates tasks when confidence is high
- 📝 **Natural language** – Uses Todoist Quick Add syntax (dates, priorities, labels)
- 🔒 **Dry-run mode** – Preview tasks before creation
- ⚡ **Fast API** – RESTful endpoint with JSON responses

## Prerequisites

- Python 3.8+
- OpenAI API key (with GPT-4 Vision access)
- Todoist API token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Codeebk/screenshot-to-todoist-task.git
cd screenshot-to-todoist-task
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install fastapi uvicorn openai requests python-multipart
```

## Configuration

Set the following environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export TODOIST_TOKEN="your-todoist-token"
export OPENAI_MODEL="gpt-4o-mini"  # Optional, defaults to gpt-4o-mini
export CONF_THRESHOLD="0.70"        # Optional, confidence threshold (0-1)
```

### Getting your Todoist token:
1. Go to [Todoist Settings → Integrations](https://todoist.com/prefs/integrations)
2. Scroll to "API token" and copy it

## Usage

### Start the server

```bash
uvicorn app:app --host 127.0.0.1 --port 8787
```

### Create a task from a screenshot

```bash
curl -X POST "http://127.0.0.1:8787/ingest" \
  -F "image=@/path/to/screenshot.png"
```

### Dry-run mode (preview without creating)

```bash
curl -X POST "http://127.0.0.1:8787/ingest?dry_run=true" \
  -F "image=@/path/to/screenshot.png"
```

### Health check

```bash
curl http://127.0.0.1:8787/health
```

## API Response Examples

### Task created successfully
```json
{
  "ok": true,
  "created": true,
  "quick_add": "Email Bryan about Q1 roadmap tomorrow p2 @work",
  "parsed": {
    "should_create_task": true,
    "quick_add": "Email Bryan about Q1 roadmap tomorrow p2 @work",
    "notes": "Mentioned in Slack thread about planning",
    "confidence": 0.95
  },
  "task": {
    "id": "12345",
    "content": "Email Bryan about Q1 roadmap"
  }
}
```

### Low confidence / needs review
```json
{
  "ok": true,
  "created": false,
  "needs_review": true,
  "threshold": 0.70,
  "parsed": {
    "should_create_task": true,
    "confidence": 0.65,
    "quick_add": "Review document p2 @inbox"
  }
}
```

### Not a task
```json
{
  "ok": true,
  "created": false,
  "parsed": {
    "should_create_task": false,
    "reason": "Screenshot shows a completed status page with no action items",
    "confidence": 0.90
  }
}
```

## How it Works

1. **Upload** – POST a screenshot to `/ingest`
2. **Analyze** – GPT-4 Vision extracts task details using structured output
3. **Validate** – Checks confidence threshold and normalizes Quick Add format
4. **Create** – Submits to Todoist Quick Add API with priority, date, and labels
5. **Respond** – Returns created task details or reason for skipping

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key for vision model |
| `TODOIST_TOKEN` | *required* | Todoist API token |
| `OPENAI_MODEL` | `gpt-4o-mini` | Vision model to use |
| `CONF_THRESHOLD` | `0.70` | Minimum confidence to auto-create (0-1) |

## Development

### Expose locally with cloudflared

```bash
brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8787
```

This creates a public HTTPS URL for testing webhooks or mobile shortcuts.

## Task Formatting

The service uses Todoist Quick Add syntax:
- **Priorities**: `p1` (urgent), `p2` (normal), `p3` (low)
- **Labels**: `@work`, `@personal`, `@inbox`
- **Dates**: `tomorrow`, `Friday 3pm`, `Jan 25`

Example: `Email client about proposal tomorrow p2 @work`

## License

MIT

## Contributing

Issues and PRs welcome! This is a simple proof-of-concept but can be extended with:
- Project mapping (auto-assign to Todoist projects)
- Multiple task extraction
- Webhook support for automation platforms
- Mobile shortcut integration
