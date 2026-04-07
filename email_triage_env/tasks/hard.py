from typing import List


def grade(actions: List[dict], ground_truth: List[dict]) -> float:
    """
    Grade full triage (label + action)
    """

    score = 0
    total = 0

    for action in actions:
        if action.get("type") == "triage":
            email_id = action.get("email_id")

            for email in ground_truth:
                if email["id"] == email_id:
                    total += 1

                    label_correct = action.get("label") == email["label"]

                    expected_action = "reply" if email["label"] == "important" else "archive"
                    action_correct = action.get("action") == expected_action

                    if label_correct:
                        score += 0.5
                    if action_correct:
                        score += 0.5

    if total == 0:
        return 0.0

    return score / total