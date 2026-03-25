from openai import OpenAI
import json

client = OpenAI()

EXTRACT_PROMPT = """
You are an expert meeting analyst. Extract ALL decisions and action items from
this transcript. Return ONLY valid JSON (no markdown, no explanation).

Format:
{
  "decisions": [
    {"description": "...", "context": "brief quote or context"}
  ],
  "action_items": [
    {"description": "...", "owner": "Person Name or Unknown", "due_date": "YYYY-MM-DD or null"}
  ]
}

Transcript:
{transcript}
"""

def extract_actions(transcript_text: str) -> dict:
    # Truncate very long transcripts to avoid token limits
    text = transcript_text[:12000]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(transcript=text)}],
            temperature=0.1,
            response_format={"type": "json_object"}   # forces valid JSON
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        # Graceful fallback on API errors
        return {"decisions": [], "action_items": [], "error": str(e)}