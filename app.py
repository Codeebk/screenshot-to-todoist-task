import os
import base64
import json
import uuid
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

# Sync + Upload endpoints (used for attaching screenshots)
TODOIST_SYNC_URL = "https://api.todoist.com/sync/v9/sync"
TODOIST_UPLOAD_URL = "https://api.todoist.com/sync/v9/uploads/add"

# Toggle screenshot attachment behavior
ATTACH_SCREENSHOT = os.environ.get("ATTACH_SCREENSHOT", "true").lower() == "true"

# Choose a vision-capable model. (You can change this any time.)
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

    # Date handling to avoid incorrectly setting task due dates
    event_datetime: Optional[str] = Field(
        default=None,
        description=(
            "Date/time mentioned in the image that refers to an event/appointment/"
            "reservation/meeting (NOT a task deadline)."
        ),
    )
    set_due: bool = Field(
        default=False,
        description="Whether to set a Todoist due date for the task.",
    )
    due_string: Optional[str] = Field(
        default=None,
        description="Todoist natural language due date (only if set_due=true). Example: 'tomorrow 3pm'.",
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

IMPORTANT: Dates and times in the image are often EVENT dates, not TASK deadlines.

You must classify any detected date/time into one of these:
A) event_datetime (appointment/reservation/meeting/showtime/etc.)
B) task_due (the deadline for when the user must complete the task)

Rules for date/time handling:
- If the user needs to do something BEFORE an event (e.g. “make a reservation for Saturday”):
  - event_datetime = "Saturday"
  - set_due = false
  - include the event date in notes
  - DO NOT put Saturday in due_string
- Only set set_due=true when the image explicitly indicates a task deadline, such as:
  - “by Friday”
  - “due tomorrow”
  - “remind me at 3pm”
  - “submit before 5pm”
  - “needs to be done today”
  - “follow up on Tuesday” (and it’s clearly a deadline for the action)
- If the message says the task is already complete (“reservation made”), do not create a follow-up task
  unless there is a next action.

Quick Add requirements:
- Format: "<verb-led task title> p2 @inbox"
- Keep it short (max ~120 chars).
- Use p1 only if urgent/production/sev/outage is implied; otherwise p2.
- Use @work if work-related, otherwise @personal.
- If the image contains specific proper nouns / field names / ticket IDs, include them in the task title.
- Do NOT invent details that aren't in the image.
- If there is a task deadline (NOT an event date), set set_due=true and put it in due_string.
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


def todoist_upload_file(file_name: str, file_bytes: bytes, content_type: str) -> dict:
    """
    Uploads a file to Todoist and returns uploaded file metadata.
    """
    if not TODOIST_TOKEN:
        raise HTTPException(status_code=500, detail="Missing TODOIST_TOKEN env var")

    headers = {"Authorization": f"Bearer {TODOIST_TOKEN}"}
    data = {"file_name": file_name}
    files = {"file": (file_name, file_bytes, content_type)}

    r = requests.post(TODOIST_UPLOAD_URL, headers=headers, data=data, files=files, timeout=30)
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Todoist upload error: {r.status_code} {r.text}")

    return r.json()


def todoist_add_item_note_with_attachment(item_id: str, content: str, file_attachment: dict) -> dict:
    """
    Adds an item note (comment) to a task with a file attachment.
    """
    if not TODOIST_TOKEN:
        raise HTTPException(status_code=500, detail="Missing TODOIST_TOKEN env var")

    headers = {"Authorization": f"Bearer {TODOIST_TOKEN}"}

    commands = [
        {
            "type": "note_add",
            "temp_id": str(uuid.uuid4()),
            "uuid": str(uuid.uuid4()),
            "args": {
                "item_id": str(item_id),
                "content": content,
                "file_attachment": file_attachment,
            },
        }
    ]

    r = requests.post(
        TODOIST_SYNC_URL,
        headers=headers,
        data={"commands": json.dumps_toggle(commands) if False else json.dumps(commands)},  # explicit json.dumps
        timeout=30,
    )

    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Todoist note_add error: {r.status_code} {r.text}")

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


def apply_due_to_quick_add(quick_add: str, due_string: Optional[str], set_due: bool) -> str:
    """
    Only attach a due date string if the model explicitly marked it as a task due date.
    This prevents event dates (appointments/reservations) from becoming task due dates.
    """
    if not set_due or not due_string:
        return quick_add
    return f"{quick_add} {due_string}".strip()


def append_event_to_notes(notes: Optional[str], event_datetime: Optional[str]) -> Optional[str]:
    if not event_datetime:
        return notes
    extra = f"Event time mentioned: {event_datetime}"
    if notes:
        return f"{notes}\n\n{extra}"
    return extra


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

    # Apply due date ONLY if it is explicitly a task deadline (not an event date)
    if parsed.quick_add:
        parsed.quick_add = apply_due_to_quick_add(parsed.quick_add, parsed.due_string, parsed.set_due)

    # Always preserve event datetime as context (notes), not as due date
    parsed.notes = append_event_to_notes(parsed.notes, parsed.event_datetime)

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

    # Attach original screenshot as a comment attachment (non-blocking)
    attachment_result = None
    if ATTACH_SCREENSHOT:
        try:
            upload_name = image.filename or "screenshot.png"
            uploaded = todoist_upload_file(upload_name, image_bytes, content_type)

            attachment_note = "📎 Original screenshot (from screenshot-to-todoist)"
            attachment_result = todoist_add_item_note_with_attachment(
                item_id=task["id"],
                content=attachment_note,
                file_attachment=uploaded,
            )
        except Exception as e:
            attachment_result = {"error": str(e)}

    return JSONResponse(
        {
            "ok": True,
            "created": True,
            "quick_add": parsed.quick_add,
            "parsed": parsed.model_dump(),
            "task": task,
            "attachment": attachment_result,
        }
    )
