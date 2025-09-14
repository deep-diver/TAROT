import json

weights = {
    "basic": 0.15,
    "medium": 0.15,
    "high": 0.35,
    "edge": 0.35,
}

evaluation_script_template = """
import subprocess
import json

def evaluate_code(code, test_cases):
    weights = {weights}

    passed_weight = 0.0
    total_weight = 0.0
    exec_timeout = 5

    for case in test_cases:
        label = case.get("label")
        if label is None: label = "minimal"
        weight = weights.get(label.strip(), 0.0)
        total_weight += weight

        process = subprocess.run(
            ["python3", "-c", code],
            input=case["input"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        output = process.stdout.strip()

        all_correct = True
        for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
            all_correct = all_correct and line1.strip() == line2.strip()

        if all_correct:
            passed_weight += weight

    # if total_weight == 0:
    #     return 0.0

    # weighted_success_rate = passed_weight / total_weight
    # return weighted_success_rate
    return passed_weight
"""

formatted_script = evaluation_script_template.format(
    weights=json.dumps(weights)
)

print(formatted_script)