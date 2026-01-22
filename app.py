import os
import base64
import json
import uuid
import requests
from typing import Optional, Dict, Any

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

# Create reminder-like "event tasks" when a future event is present (dinner, appointment, flight, etc.)
CREATE_EVENT_TASKS = os.environ.get("CREATE_EVENT_TASKS", "true").lower() == "true"

# Choose a vision-capable model. (You can change this any time.)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Only auto-create tasks if confidence is high enough
AUTO_CREATE_CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.70"))


# ====== Schema (Structured Output) ======
class TaskParseResult(BaseModel):
    # ACTION TASK (the thing you need to DO)
    should_create_task: bool = Field(
        description="True if the image contains a clear actionable task the user should do."
    )
    quick_add: Optional[str] = Field(
        default=None,
        description="Todoist Quick Add string for the ACTION task. Example: 'Email Bryan p2 @work'",
    )

    # Notes should be rare + only used when vital info doesn't fit in title
    notes: Optional[str] = Field(
        default=None,
        description="Extra context from the screenshot (short). Should usually be null.",
    )

    # Date handling to avoid incorrectly setting ACTION task due dates
    event_datetime: Optional[str] = Field(
        default=None,
        description=(
            "Date/time mentioned in the image that refers to an event/appointment/"
            "reservation/meeting (NOT a task deadline)."
        ),
    )
    set_due: bool = Field(
        default=False,
        description="Whether to set a Todoist due date for the ACTION task.",
    )
    due_string: Optional[str] = Field(
        default=None,
        description="Todoist natural language due date for the ACTION task (only if set_due=true).",
    )

    # EVENT REMINDER TASK (calendar-style reminder like: Dinner at X Saturday 6pm)
    create_event_task: bool = Field(
        default=False,
        description="True if a reminder task should be created for the event itself.",
    )
    event_quick_add: Optional[str] = Field(
        default=None,
        description="Todoist Quick Add string for the EVENT reminder task. Example: 'Dinner at Fonda San Miguel p3 @personal'",
    )
    event_due_string: Optional[str] = Field(
        default=None,
        description="Todoist due string for the EVENT reminder, like 'Saturday 6pm'.",
    )

    reason: Optional[str] = Field(
        default=None,
        description="If should_create_task is false, explain why (short).",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How confident you are that this parse is correct."
    )


SYSTEM_PROMPT = """
You are a task-extraction assistant.

You will be given an image (screenshot or photo). Decide if it contains:
- an ACTION ITEM the user must DO
- and/or a future EVENT that should be remembered

Rules:
- Ignore UI chrome, headers, usernames, timestamps, reactions, menus, etc.
- If it's a conversation/email/transcript, infer the most likely next action for the user.
- Prefer ONE action task. If multiple, pick the most important one.
- Do NOT hedge with "maybe", "might", "possibly". Write tasks directly.

Output must match the JSON schema exactly.

IMPORTANT: Dates/times in images are often EVENT dates, not ACTION deadlines.

Classify date/time into:
A) event_datetime (appointment/reservation/meeting/showtime/etc.)
B) action task due date (deadline)

ACTION DATE RULES:
- If the user needs to do something BEFORE an event (e.g. “make a reservation for Saturday at 6pm”):
  - event_datetime = "Saturday 6pm"
  - set_due = false
  - DO NOT put "Saturday 6pm" in due_string for the action task
- Only set set_due=true for the action task when the image explicitly indicates a deadline, such as:
  - “by Friday”
  - “due tomorrow”
  - “remind me at 3pm”
  - “submit before 5pm”
  - “needs to be done today”
  - “follow up on Tuesday” (when it's clearly a deadline for the action)

HARD RULE:
- If the ACTION task is about making/booking/reserving/planning something for a future event,
  then set_due MUST be false unless the screenshot explicitly states a deadline (e.g. "by Friday", "today", "ASAP").
- Never use the event time (event_datetime) as the due date of the ACTION task.

COMPLETED ACTIONS (very important):
- If the screenshot indicates the action is already completed (e.g., "reservation made", "booked", "confirmed", "done"):
  - Prefer should_create_task = false (informational only)
  - Only create an action task if there is a clear next action (invite people, confirm attendees, pay invoice, etc.)

EVENT REMINDER TASKS (important):
- If the screenshot mentions a future plan/event (dinner, appointment, reservation time, meeting, flight, etc.):
  - Prefer create_event_task = true
  - event_quick_add should be a reminder-style title like:
    "Dinner at Fonda San Miguel p3 @personal"
  - event_due_string should be the event time (e.g., "Saturday 6pm")
- This event reminder task should be created even if the action is already complete.
- For the EVENT reminder task, it is correct/desirable to use the event date/time as the due date.

Quick Add requirements (both action and event tasks):
- Format: "<title> p2 @inbox"
- Keep it short (max ~120 chars).
- Use p1 only if urgent/production/sev/outage is implied.
- Use @work if work-related, otherwise @personal.
- Do NOT invent details that aren't in the image.

RESERVATION ACTION TITLE RULE (hard rule):
- If the user needs to book/make a reservation, the ACTION task title MUST start with:
  "Make reservation at <place>"
  Example: "Make reservation at Fonda San Miguel"
- Do NOT use "Dinner at ..." for the ACTION task.
- The "Dinner at ..." phrasing is only for the EVENT reminder task.

TITLE VS NOTES:
- Strong preference: keep key details in the title rather than notes.
- Include key details in the title when it stays readable:
  - who/where (restaurant/company/person)
  - when (Tuesday 6pm)
  - critical identifiers (ticket ID / field name)
- notes should be NULL unless it adds vital information not captured in the title.
- If notes are included, keep them brief (1–2 sentences).
- Do NOT paste large blocks of transcript text into notes.
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
                    {"type": "input_text", "text": "Extract the best Todoist capture(s) from this image."},
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
        data={"commands": json.dumps(commands)},
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


def normalize_event_quick_add(q: str) -> str:
    """
    Event reminders should default to p3 if the model didn't specify.
    """
    q = q.strip()

    if " p1" not in q and " p2" not in q and " p3" not in q:
        q += " p3"

    if "@work" not in q and "@personal" not in q and "@inbox" not in q:
        q += " @personal"

    return q


def looks_like_same_datetime(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return a.strip().lower() == b.strip().lower()


def enforce_reservation_action_wording(action_quick_add: str) -> str:
    """
    If the model generates an ACTION task like 'Reservation at X', rewrite it to:
      'Make reservation at X'
    This ONLY affects the ACTION task, not the EVENT reminder task.
    """
    q = action_quick_add.strip()
    lower = q.lower()

    # Only rewrite if it looks like an ACTION reservation (not already "make reservation")
    if lower.startswith("reservation at ") and "make reservation" not in lower:
        q = "Make " + q  # -> "Make Reservation at ..."
        q = q.replace("Make Reservation", "Make reservation", 1)

    return q


def apply_due_to_action_quick_add(quick_add: str, due_string: Optional[str], set_due: bool) -> str:
    """
    Only attach a due date string if the model explicitly marked it as a TASK deadline.
    This prevents event dates (appointments/reservations) from becoming task due dates.
    """
    if not set_due or not due_string:
        return quick_add
    return f"{quick_add} {due_string}".strip()


def build_event_quick_add_with_due(event_quick_add: str, event_due_string: Optional[str]) -> str:
    """
    For event reminder tasks, it is correct to include the event time as the due date.
    """
    if event_due_string:
        return f"{event_quick_add} {event_due_string}".strip()
    return event_quick_add


def append_event_to_notes(notes: Optional[str], event_datetime: Optional[str]) -> Optional[str]:
    """
    Preserve event time as context, but do NOT treat it like a task due date.
    Only add it to notes if it exists.
    """
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

    # ===== Guardrail: never let ACTION due date equal EVENT time =====
    # If an event reminder exists, the event time belongs ONLY on the event task.
    if parsed.create_event_task and parsed.event_due_string and parsed.set_due and parsed.due_string:
        if looks_like_same_datetime(parsed.due_string, parsed.event_due_string):
            parsed.set_due = False
            parsed.due_string = None

    # ===== Normalize + build final strings =====
    action_text: Optional[str] = None
    event_text: Optional[str] = None

    if parsed.quick_add:
        parsed.quick_add = normalize_quick_add(parsed.quick_add)
        parsed.quick_add = enforce_reservation_action_wording(parsed.quick_add)
        parsed.quick_add = apply_due_to_action_quick_add(parsed.quick_add, parsed.due_string, parsed.set_due)
        action_text = parsed.quick_add

    if parsed.event_quick_add:
        parsed.event_quick_add = normalize_event_quick_add(parsed.event_quick_add)
        event_text = build_event_quick_add_with_due(parsed.event_quick_add, parsed.event_due_string)

    # Preserve event datetime as context (notes), not as due date
    parsed.notes = append_event_to_notes(parsed.notes, parsed.event_datetime)

    # Only send notes if non-empty
    note_to_send = parsed.notes if (parsed.notes and parsed.notes.strip()) else None

    # Determine whether we will create anything
    will_create_action = bool(parsed.should_create_task and action_text)
    will_create_event = bool(CREATE_EVENT_TASKS and parsed.create_event_task and event_text)

    # If nothing to do, return cleanly
    if not will_create_action and not will_create_event:
        return JSONResponse({"ok": True, "created": False, "parsed": parsed.model_dump()})

    # Safety rail: require confidence
    if parsed.confidence < AUTO_CREATE_CONFIDENCE_THRESHOLD:
        return JSONResponse(
            {
                "ok": True,
                "created": False,
                "needs_review": True,
                "threshold": AUTO_CREATE_CONFIDENCE_THRESHOLD,
                "parsed": parsed.model_dump(),
                "preview": {
                    "will_create_action": will_create_action,
                    "action_text": action_text,
                    "will_create_event": will_create_event,
                    "event_text": event_text,
                },
            }
        )

    # Dry run: show what WOULD happen
    if dry_run:
        return JSONResponse(
            {
                "ok": True,
                "created": False,
                "parsed": parsed.model_dump(),
                "preview": {
                    "will_create_action": will_create_action,
                    "action_text": action_text,
                    "will_create_event": will_create_event,
                    "event_text": event_text,
                },
            }
        )

    # ===== Create tasks =====
    action_task: Optional[Dict[str, Any]] = None
    event_task: Optional[Dict[str, Any]] = None

    if will_create_action and action_text:
        action_task = todoist_quick_add(text=action_text, note=note_to_send)

    if will_create_event and event_text:
        # event tasks typically don't need notes (keep clean)
        event_task = todoist_quick_add(text=event_text, note=None)

    # ===== Attach original screenshot to any created task(s) =====
    attachments: Dict[str, Any] = {"upload": None, "action_note": None, "event_note": None}
    if ATTACH_SCREENSHOT and (action_task or event_task):
        try:
            upload_name = image.filename or "screenshot.png"
            uploaded = todoist_upload_file(upload_name, image_bytes, content_type)
            attachments["upload"] = {"ok": True, "file_name": upload_name}

            attachment_note_text = "📎 Original screenshot (from screenshot-to-todoist)"

            if action_task:
                attachments["action_note"] = todoist_add_item_note_with_attachment(
                    item_id=action_task["id"],
                    content=attachment_note_text,
                    file_attachment=uploaded,
                )

            if event_task:
                attachments["event_note"] = todoist_add_item_note_with_attachment(
                    item_id=event_task["id"],
                    content=attachment_note_text,
                    file_attachment=uploaded,
                )

        except Exception as e:
            attachments["upload"] = {"ok": False, "error": str(e)}

    return JSONResponse(
        {
            "ok": True,
            "created": True,
            "parsed": parsed.model_dump(),
            "created_action_task": bool(action_task),
            "created_event_task": bool(event_task),
            "action_text": action_text,
            "event_text": event_text,
            "action_task": action_task,
            "event_task": event_task,
            "attachments": attachments,
        }
    )
