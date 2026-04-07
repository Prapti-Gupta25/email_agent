from typing import List


def grade(actions: List[dict], ground_truth: List[dict]) -> float:

    correct = 0
    total = 0


    for action in actions:
        if action.get("type") == "classify":
                email_id = action.get("email_id")
                label = action.get("label")

                for email in ground_truth:
                     if email["id"] == email_id:
                          total += 1
                          if label == email["label"]:
                               correct += 1

    if total == 0:
         return 0.0
    
    return correct/total