import os
import sys
from openai import OpenAI
from server.drug_dosage_env_environment import DrugDosageEnvironment
from models import DrugDosageAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_dummy"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "drug_dosage_env"
MAX_STEPS = 5

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a clinical AI assistant helping doctors choose the correct drug and dosage.
You will receive patient information and must respond with ONLY a JSON object like this:
{"drug_name": "paracetamol", "dosage_mg": 500, "route": "oral"}
Choose only from the available_drugs list. Be careful about allergies and kidney function."""

def run_task(task_name: str):
    env = DrugDosageEnvironment(task_name=task_name)
    obs = env.reset()

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    rewards = []
    step = 0

    for step in range(1, MAX_STEPS + 1):
        # Build prompt from observation
        user_prompt = f"""Patient Information:
- Age: {obs.patient_age} years
- Weight: {obs.patient_weight_kg} kg
- Condition: {obs.condition}
- Allergies: {obs.allergies if obs.allergies else 'None'}
- Blood Pressure: {obs.blood_pressure}
- Kidney Function: {obs.kidney_function}
- Available Drugs: {obs.available_drugs}

What drug and dosage do you recommend? Respond only with JSON."""

        error_msg = "null"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=150,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON response from model
            import json
            # Clean markdown fences if model adds them
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            action = DrugDosageAction(
                drug_name=data.get("drug_name", "unknown"),
                dosage_mg=float(data.get("dosage_mg", 0)),
                route=data.get("route", "oral"),
            )
            action_str = f"drug={action.drug_name},dose={action.dosage_mg}mg"

        except Exception as e:
            error_msg = str(e).replace("\n", " ")[:100]
            action = DrugDosageAction(drug_name="unknown", dosage_mg=0, route="oral")
            action_str = "parse_error"

        obs = env.step(action)
        rewards.append(obs.reward)

        print(f"[STEP] step={step} action={action_str} reward={obs.reward:.2f} done={str(obs.done).lower()} error={error_msg}")

        if obs.done:
            break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success = rewards[-1] >= 0.8 if rewards else False
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")
    print()

if __name__ == "__main__":
    for task in ["very_easy", "easy", "medium", "hard", "very_hard"]:
        run_task(task)