import os
import json
from openai import OpenAI

from email_triage_env.server.email_triage_env_environment import EmailTriageEnvironment
from email_triage_env.models import EmailTriageAction


API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

MAX_STEPS = 5

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def format_emails(observation):
    return "\n".join([
        f"ID: {e.id}, Subject: {e.subject}, Body: {e.body}, Sender: {e.sender}"
        for e in observation.emails
    ])


def get_classify_action(observation, remaining_ids):
    target_id = remaining_ids[0]
    prompt = f"""You are an email assistant. Classify email with ID {target_id} only.
Return ONLY valid JSON (no extra text):
{{
  "type": "classify",
  "email_id": {target_id},
  "label": "spam" or "important" or "normal"
}}

Emails:
{format_emails(observation)}"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = completion.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        response = json.loads(content)
        if isinstance(response, list):
            response = response[0]
        return EmailTriageAction(**response)
    except Exception as e:
        print(f"  [classify error] {e}")
        return EmailTriageAction(type="classify", email_id=target_id, label="normal")


def get_prioritize_action(observation, remaining_ids):
    target_id = remaining_ids[0]
    ids = [e.id for e in observation.emails]
    prompt = f"""You are an email assistant. Order these email IDs by priority — most important first.

Return ONLY valid JSON (no extra text):
{{
  "type": "prioritize",
  "email_id": {target_id},
  "order": {ids}
}}

Reorder the list in 'order' — put urgent/work emails first, spam last.

Emails:
{format_emails(observation)}"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = completion.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        response = json.loads(content)
        if isinstance(response, list):
            response = response[0]
        return EmailTriageAction(**response)
    except Exception as e:
        print(f"  [prioritize error] {e}")
        return EmailTriageAction(type="prioritize", email_id=target_id, order=ids)


def get_triage_action(observation, remaining_ids):
    target_id = remaining_ids[0]
    prompt = f"""You are an email assistant. For email with ID {target_id}, classify it AND decide the action.

Return ONLY valid JSON (no extra text):
{{
  "type": "triage",
  "email_id": {target_id},
  "label": "spam" or "important" or "normal",
  "action": "reply" or "archive"
}}

Rules:
- important -> action should be "reply"
- spam or normal -> action should be "archive"

Emails:
{format_emails(observation)}"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = completion.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        response = json.loads(content)
        if isinstance(response, list):
            response = response[0]
        return EmailTriageAction(**response)
    except Exception as e:
        print(f"  [triage error] {e}")
        return EmailTriageAction(type="triage", email_id=target_id, label="normal", action="archive")


def run_task(task_name, get_action_fn):
    env = EmailTriageEnvironment()
    observation = env.reset()
    rewards = []
    done = False
    step = 0

    remaining_ids = [e.id for e in observation.emails]

    print(f"[START] task={task_name} env=email_triage model={MODEL_NAME}")

    while not done and step < MAX_STEPS and remaining_ids:
        step += 1

        action = get_action_fn(observation, remaining_ids)
        observation = env.step(action)

        reward = observation.reward
        done = observation.done
        rewards.append(f"{reward:.2f}")

        if action.email_id in remaining_ids:
            remaining_ids.remove(action.email_id)

        if not remaining_ids:
            done = True

        print(f"[STEP] step={step} action={action.model_dump()} reward={reward:.2f} done={str(done).lower()} error=null")

    score = sum(float(r) for r in rewards) / len(rewards) if rewards else 0.0
    success = "true" if any(float(r) > 0 for r in rewards) else "false"

    print(f"[END] success={success} steps={step} score={score:.2f} rewards={','.join(rewards)}")
    print()
    return score


def run():
    scores = {}

    scores["easy"]   = run_task("easy",   get_classify_action)
    scores["medium"] = run_task("medium", get_prioritize_action)
    scores["hard"]   = run_task("hard",   get_triage_action)

    overall = sum(scores.values()) / len(scores)
    print(f"=== FINAL SCORES ===")
    print(f"Easy:   {scores['easy']:.2f}")
    print(f"Medium: {scores['medium']:.2f}")
    print(f"Hard:   {scores['hard']:.2f}")
    print(f"Overall: {overall:.2f}")


if __name__ == "__main__":
    run()