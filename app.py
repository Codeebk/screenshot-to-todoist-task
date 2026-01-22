import os
import base64
import requests
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI


# ====== Config ======
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
TODOIST_TOKEN = (os.environ.get("TODOIST_TOKEN") or "").strip()


# Todoist Quick Add (Sync API v9) – deprecated but still the easiest way to parse natural language
# You can swap this later if Todoist changes it.
TODOIST_QUICK_ADD_URL = "https://api.todoist.com/sync/v9/quick/add"

# Choose a vision-capable model. (You can change this any time.)
# The OpenAI API supports image analysis through the Responses API. :contentReference[oaicite:2]{index=2}
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Only auto-create tasks if confidence is high enough
AUTO_CREATE_CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.70"))


# ====== Schema (Structured Output) ======
class TaskParseResult(BaseModel):
    should_create_task: bool = Field(
        description="True if the image contains a clear actionable task the user should do."
    )
    quick_add: Optional[str] = Field(
        default=None,
        description="Todoist Quick Add string. Example: 'Email Bryan tomorrow p2 @work'",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Extra context from the screenshot (short).",
    )
    reason: Optional[str] = Field(
        default=None,
        description="If should_create_task is false, explain why (short).",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How confident you are that this is a correct task."
    )


SYSTEM_PROMPT = """
You are a task-extraction assistant.

You will be given an image (screenshot or photo). Decide if it contains an ACTION ITEM for the user.

Rules:
- Only output a task if the user needs to DO something.
- Ignore UI chrome, headers, usernames, timestamps, reactions, menus, etc.
- If it's a conversation/email/transcript, infer the most likely next action for the user.
- Prefer ONE task. If multiple, pick the most important one.
- Do NOT hedge with "maybe", "might", "possibly". Write the task directly.

Output must match the JSON schema exactly.

Quick Add requirements:
- Format: "<verb-led task title> p2 @inbox"
- Keep it short (max ~120 chars).
- If there is a time/date in the image, include it (e.g., "tomorrow", "Friday 3pm").
- Use p1 only if urgent/production/sev/outage is implied; otherwise p2.
- Use @work if work-related, otherwise @personal.
- If the image contains specific proper nouns / field names / ticket IDs, include them in the task title.
- Do NOT invent details that aren't in the image.
"""



app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)


def to_data_url(image_bytes: bytes, content_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def parse_task_from_image(image_bytes: bytes, content_type: str) -> TaskParseResult:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY env var")

    data_url = to_data_url(image_bytes, content_type)

    # Structured output parsing (schema-locked)
    # OpenAI Structured Outputs guide: ensures schema adherence. :contentReference[oaicite:3]{index=3}
    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the single best Todoist task from this image."},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=TaskParseResult,
    )

    return resp.output_parsed


def todoist_quick_add(text: str, note: Optional[str] = None) -> dict:
    if not TODOIST_TOKEN:
        raise HTTPException(status_code=500, detail="Missing TODOIST_TOKEN env var")

    headers = {"Authorization": f"Bearer {TODOIST_TOKEN}"}
    data = {"text": text}
    if note:
        data["note"] = note

    r = requests.post(TODOIST_QUICK_ADD_URL, headers=headers, data=data, timeout=20)
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Todoist error: {r.status_code} {r.text}")

    return r.json()

def normalize_quick_add(q: str) -> str:
    q = q.strip()

    # Ensure there's a priority
    if " p1" not in q and " p2" not in q and " p3" not in q:
        q += " p2"

    # Ensure there's a label
    if "@work" not in q and "@personal" not in q and "@inbox" not in q:
        q += " @inbox"

    return q



@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
async def ingest(image: UploadFile = File(...), dry_run: bool = False):
    image_bytes = await image.read()
    content_type = image.content_type or "image/png"

    parsed = parse_task_from_image(image_bytes, content_type)

    # Normalize early so dry_run also shows the final task formatting
    if parsed.quick_add:
        parsed.quick_add = normalize_quick_add(parsed.quick_add)

    if not parsed.should_create_task:
        return JSONResponse({"ok": True, "created": False, "parsed": parsed.model_dump()})

    if parsed.confidence < AUTO_CREATE_CONFIDENCE_THRESHOLD:
        return JSONResponse(
            {
                "ok": True,
                "created": False,
                "needs_review": True,
                "threshold": AUTO_CREATE_CONFIDENCE_THRESHOLD,
                "parsed": parsed.model_dump(),
            }
        )

    if dry_run:
        return JSONResponse({"ok": True, "created": False, "parsed": parsed.model_dump()})

    if not parsed.quick_add:
        raise HTTPException(
            status_code=422,
            detail="Model returned should_create_task=true but quick_add was empty",
        )

    task = todoist_quick_add(text=parsed.quick_add, note=parsed.notes)

    return JSONResponse(
        {
            "ok": True,
            "created": True,
            "quick_add": parsed.quick_add,
            "parsed": parsed.model_dump(),
            "task": task,
        }
    )

