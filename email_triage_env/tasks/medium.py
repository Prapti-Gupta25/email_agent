def grade(action: dict, ground_truth: list) -> float:
    """
    Grade ordering correctness
    """

    correct_order = sorted(
        ground_truth,
        key=lambda x: x["label"] == "important",
        reverse=True,
    )

    correct_ids = [e["id"] for e in correct_order]
    predicted = action.get("order", [])

    if not predicted:
        return 0.0

    matches = sum(
        1 for i in range(min(len(predicted), len(correct_ids)))
        if predicted[i] == correct_ids[i]
    )

    return matches / len(correct_ids)