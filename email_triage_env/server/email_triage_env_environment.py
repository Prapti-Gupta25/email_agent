import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from email_triage_env.models import EmailTriageAction, EmailTriageObservation, Email

EMAIL_POOL = [
    {"id": 1, "subject": "Urgent: Server down", "body": "Production is failing, fix ASAP", "sender": "cto@company.com", "label": "important"},
    {"id": 2, "subject": "Meeting tomorrow at 10 AM", "body": "Please attend the standup", "sender": "boss@company.com", "label": "important"},
    {"id": 3, "subject": "Win a million dollars!!!", "body": "Click this link now to claim prize", "sender": "spam@fake.com", "label": "spam"},
    {"id": 4, "subject": "You have been selected!", "body": "Congratulations! Send your details", "sender": "lottery@scam.net", "label": "spam"},
    {"id": 5, "subject": "Lunch?", "body": "Are you free today for lunch?", "sender": "friend@gmail.com", "label": "normal"},
    {"id": 6, "subject": "Weekend plans", "body": "Hey, what are you doing this weekend?", "sender": "colleague@gmail.com", "label": "normal"},
    {"id": 7, "subject": "Project deadline extended", "body": "The deadline is now next Friday", "sender": "manager@company.com", "label": "important"},
    {"id": 8, "subject": "Free iPhone giveaway", "body": "You won! Click to claim your iPhone", "sender": "promo@fake.biz", "label": "spam"},
    {"id": 9, "subject": "Team outing this Saturday", "body": "Join us for a team lunch at 1 PM", "sender": "hr@company.com", "label": "normal"},
    {"id": 10, "subject": "Invoice overdue", "body": "Your payment of $500 is overdue", "sender": "billing@vendor.com", "label": "important"},
    {"id": 11, "subject": "Password reset request", "body": "Someone requested a password reset", "sender": "security@company.com", "label": "important"},
    {"id": 12, "subject": "Claim your free gift", "body": "Limited time offer, click now!", "sender": "offers@spam.net", "label": "spam"},
    {"id": 13, "subject": "Happy birthday!", "body": "Wishing you a wonderful birthday", "sender": "colleague@gmail.com", "label": "normal"},
    {"id": 14, "subject": "System maintenance tonight", "body": "Servers will be down from 2-4 AM", "sender": "it@company.com", "label": "important"},
    {"id": 15, "subject": "You won a lucky draw", "body": "Send your bank details to claim", "sender": "lucky@scam.biz", "label": "spam"},
    {"id": 16, "subject": "Client complaint received", "body": "Urgent: client is unhappy with delivery", "sender": "support@company.com", "label": "important"},
    {"id": 17, "subject": "Movie tonight?", "body": "Want to catch a movie after work?", "sender": "friend@gmail.com", "label": "normal"},
    {"id": 18, "subject": "Your account will be suspended", "body": "Verify your details immediately", "sender": "noreply@phishing.com", "label": "spam"},
    {"id": 19, "subject": "Quarterly review scheduled", "body": "Your review is on Friday at 3 PM", "sender": "hr@company.com", "label": "important"},
    {"id": 20, "subject": "Weekend cricket match", "body": "Join us for cricket on Sunday!", "sender": "team@gmail.com", "label": "normal"},

]


class EmailTriageEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.emails = []
        self.max_steps = 5


    def reset(self) -> EmailTriageObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.emails = random.sample(EMAIL_POOL, 3)

        for i, email in enumerate(self.emails):
            email = dict(email)
            email["id"] = i + 1
            self.emails[i] = email

        return EmailTriageObservation(
            emails=[Email(id=e["id"], subject=e["subject"], body=e["body"], sender=e["sender"]) for e in self.emails],
            step_count=0,
            done=False,
            reward=0.0,
        )

    
    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        self._state.step_count += 1

        reward = 0.0
        done = False

        #easy
        if action.type == "classify":
            for email in self.emails:
                if email["id"] == action.email_id:
                    if action.label == email["label"]:
                        reward = 1.0
                    elif action.label in ["spam", "important", "normal"]:
                        # Partial credit — wrong label but valid guess
                        reward = 0.2
                    else:
                        reward = 0.0

        #medium
        elif action.type == "prioritize":
            correct_order = [e["id"] for e in sorted(self.emails, key=lambda x: (x["label"] != "important", x["label"] != "normal"))]

            if action.order:
                matches = sum(1 for i, eid in enumerate(action.order) if i < len(correct_order) and eid == correct_order[i])
                reward = round(matches / len(correct_order), 2)

        #hard
        elif action.type == "triage":
            for email in self.emails:
                if email["id"] == action.email_id:
                    label_correct = action.label == email["label"]
                    expected_action = "reply" if email["label"] == "important" else "archive"
                    action_correct = action.action == expected_action

                    if label_correct and action_correct:
                        reward = 1.0        
                    elif label_correct or action_correct:
                        reward = 0.5       
                    else:
                        reward = 0.0        

       
        self.emails = [e for e in self.emails if e["id"] != action.email_id]

        if self._state.step_count >= self.max_steps or len(self.emails) == 0:
            done = True

        return EmailTriageObservation(
            emails=[Email(id=e["id"], subject=e["subject"], body=e["body"], sender=e["sender"]) for e in self.emails],
            step_count=self._state.step_count,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state