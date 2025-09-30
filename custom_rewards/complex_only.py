import re
import json
import asyncio
from .utils import is_e2b_available

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None

def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer

def run_async_from_sync(scripts: list[str], language: str) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards

async def run_async(scripts: list[str], language: str) -> list[float]:
    # Create the sandbox by hand, currently there's no context manager for this version
    sbx = await AsyncSandbox.create(timeout=60, request_timeout=10)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(sbx, script, language) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    # Kill the sandbox after all the tasks are complete
    await sbx.kill()

    return rewards

async def run_script(sbx: AsyncSandbox, script: str, language: str) -> float:
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0
    except Exception as e:
        print(f"Error from E2B executor run_script: {e}")
        return 0.0

def code_based_on_unittests_reward(completions, **kwargs) -> list[float]:
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # Returns a reward function that evaluates code snippets in a sandbox.
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        weights = {{
            # "basic": 1.0,
            # "medium": 0.25,
            "high": 1.0,
            # "edge": 1.0,
        }}

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

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """        

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]    
    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)
    try:
        rewards = run_async_from_sync(scripts, language)

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards

# Your curriculum logic
def label_schedule(epoch: int):
    # if step < 100:
    #     return None
    # if epoch < 0.2:
    #     return ["edge"]
    # elif epoch < 0.4:
    #     return ["high", "edge"]
    # elif epoch < 0.6:
    #     return ["medium", "high", "edge"]
    # else:
    #     return ["basic", "medium", "high", "edge"]
    return ["high"]
    # if epoch < 0.2:
    #     return ["basic"]
    # elif epoch < 0.4:
    #     return ["basic", "medium"]
    # elif epoch < 0.6:
    #     return ["basic", "medium", "high"]
    # else:
    #     return ["basic", "medium", "high", "edge"]

def curriculum_aware_reward_fn(completions, **kwargs) -> list[float]:
    current_epoch = kwargs["epoch"]
    print(f"Current epoch: {current_epoch}")
    allowed_labels = label_schedule(current_epoch)
    print(f"Allowed labels: {allowed_labels}")
    # if allowed_labels is None:
    #     return [None] * len(completions)

    for info in kwargs["verification_info"]:
        info["test_cases"] = [
            case for case in info["test_cases"]
            if (case.get("label") or "minimal") in allowed_labels
        ]

    rewards = code_based_on_unittests_reward(completions, **kwargs)
    return rewards
