from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DrugDosageAction, DrugDosageObservation
except ImportError:
    from models import DrugDosageAction, DrugDosageObservation

DRUG_RULES = {
    "paracetamol": {
        "standard_dose_mg": 500,
        "max_dose_mg": 1000,
        "treats": ["fever", "mild pain"],
        "contraindications": ["liver disease"],
        "renal_adjustment": False,
    },
    "amoxicillin": {
        "standard_dose_mg": 500,
        "max_dose_mg": 1000,
        "treats": ["bacterial infection", "pneumonia"],
        "contraindications": ["penicillin allergy"],
        "renal_adjustment": True,
    },
    "ibuprofen": {
        "standard_dose_mg": 400,
        "max_dose_mg": 800,
        "treats": ["inflammation", "mild pain", "fever"],
        "contraindications": ["kidney disease", "stomach ulcer"],
        "renal_adjustment": True,
    },
    "metformin": {
        "standard_dose_mg": 500,
        "max_dose_mg": 1000,
        "treats": ["diabetes"],
        "contraindications": ["kidney disease"],
        "renal_adjustment": True,
    },
    "adrenaline": {
        "standard_dose_mg": 0.5,
        "max_dose_mg": 1.0,
        "treats": ["anaphylaxis", "cardiac arrest"],
        "contraindications": [],
        "renal_adjustment": False,
    },
}

TASKS = {
    "very_easy": {
        "patient_age": 8,
        "patient_weight_kg": 25,
        "condition": "mild pain",
        "allergies": [],
        "blood_pressure": "100/65",
        "kidney_function": "normal",
        "available_drugs": ["paracetamol", "ibuprofen", "amoxicillin"],
        "correct_drug": "paracetamol",
        "correct_dose_mg": 250,
        "dose_tolerance_mg": 100,
    },
    "easy": {
        "patient_age": 30,
        "patient_weight_kg": 70,
        "condition": "fever",
        "allergies": [],
        "blood_pressure": "120/80",
        "kidney_function": "normal",
        "available_drugs": ["paracetamol", "ibuprofen", "amoxicillin"],
        "correct_drug": "paracetamol",
        "correct_dose_mg": 500,
        "dose_tolerance_mg": 200,
    },
    "medium": {
        "patient_age": 55,
        "patient_weight_kg": 80,
        "condition": "bacterial infection",
        "allergies": ["penicillin allergy"],
        "blood_pressure": "140/90",
        "kidney_function": "impaired",
        "available_drugs": ["amoxicillin", "ibuprofen", "paracetamol"],
        "correct_drug": "paracetamol",
        "correct_dose_mg": 500,
        "dose_tolerance_mg": 150,
    },
    "hard": {
        "patient_age": 45,
        "patient_weight_kg": 65,
        "condition": "anaphylaxis",
        "allergies": [],
        "blood_pressure": "70/40",
        "kidney_function": "normal",
        "available_drugs": ["adrenaline", "paracetamol", "metformin", "ibuprofen"],
        "correct_drug": "adrenaline",
        "correct_dose_mg": 0.5,
        "dose_tolerance_mg": 0.3,
    },
    "very_hard": {
        "patient_age": 65,
        "patient_weight_kg": 72,
        "condition": "diabetes",
        "allergies": ["penicillin allergy", "stomach ulcer"],
        "blood_pressure": "160/95",
        "kidney_function": "severe",
        "available_drugs": ["metformin", "ibuprofen", "amoxicillin", "paracetamol", "adrenaline"],
        "correct_drug": "paracetamol",
        "correct_dose_mg": 500,
        "dose_tolerance_mg": 150,
    },
}


class DrugDosageEnvironment(Environment):
    """
    AI Drug Dosage Environment.
    The AI agent must identify the correct drug and safe dosage
    for a patient based on their condition, allergies, and kidney function.
    Tasks: very_easy -> easy -> medium -> hard -> very_hard
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "easy"):
        self._task_name = task_name if task_name in TASKS else "easy"
        self._task = TASKS[self._task_name]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_reward = 0.0

    def reset(self) -> DrugDosageObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_reward = 0.0
        t = self._task
        return DrugDosageObservation(
            patient_age=t["patient_age"],
            patient_weight_kg=t["patient_weight_kg"],
            condition=t["condition"],
            allergies=t["allergies"],
            blood_pressure=t["blood_pressure"],
            kidney_function=t["kidney_function"],
            available_drugs=t["available_drugs"],
            feedback="Patient admitted. Choose the correct drug and dosage.",
            task_name=self._task_name,
            done=False,
            reward=0.05,
        )

    def step(self, action: DrugDosageAction) -> DrugDosageObservation:
        self._state.step_count += 1
        t = self._task
        reward = 0.05
        feedback_parts = []

        drug = action.drug_name.lower().strip()
        dose = action.dosage_mg
        rules = DRUG_RULES.get(drug)

        # Check drug exists
        if rules is None or drug not in t["available_drugs"]:
            self._done = True
            return self._obs(0.05, "Unknown or unavailable drug selected.", done=True)

        # Check contraindications
        for contra in rules["contraindications"]:
            if contra in t["allergies"] or contra == t["condition"]:
                self._done = True
                return self._obs(0.05 , f"Dangerous! {drug} is contraindicated for this patient.", done=True)

        # Check drug treats the condition
        treats_condition = any(c in t["condition"] for c in rules["treats"])
        if not treats_condition:
            reward = 0.1
            feedback_parts.append(f"{drug} does not treat {t['condition']}.")
        else:
            reward += 0.4
            feedback_parts.append(f"Good drug choice for {t['condition']}.")

        # Renal adjustment
        if rules["renal_adjustment"] and t["kidney_function"] != "normal":
           # FIX - both lines same indent:
            reward = max(0.01, reward - 0.2)
            feedback_parts.append("Warning: dose reduction needed for impaired kidneys.")

        # Weight-based dosing
        weight = t["patient_weight_kg"]
        age = t["patient_age"]
        correct_dose = t["correct_dose_mg"]

        if age < 12:
            correct_dose = min(round(weight * 10, 0), rules["max_dose_mg"])
            feedback_parts.append(f"Child patient: weight-based dose = {correct_dose}mg ({weight}kg x 10mg/kg).")
        elif age >= 65:
            correct_dose = round(correct_dose * 0.75, 0)
            feedback_parts.append(f"Elderly patient: reduced dose = {correct_dose}mg (75% of standard).")

        # Check dosage
        tolerance = t["dose_tolerance_mg"]
        dose_diff = abs(dose - correct_dose)

        if dose > rules["max_dose_mg"]:
            reward = 0.05  # was 0.0
            feedback_parts.append(f"Overdose! Max allowed is {rules['max_dose_mg']}mg.")
        elif dose_diff <= tolerance * 0.2:
            reward += 0.6
            feedback_parts.append(f"Dosage is perfect! ({correct_dose}mg)")
        elif dose_diff <= tolerance:
            reward += 0.3
            feedback_parts.append(f"Dosage acceptable but not optimal. Ideal: {correct_dose}mg.")
        else:
            reward += 0.1
            feedback_parts.append(f"Dosage is off. Ideal: {correct_dose}mg.")

        reward = round(min(reward, 0.99), 2)  # cap at 0.99 not 1.0
        reward = max(reward, 0.01)            # floor at 0.01 not 0.0
        self._done = True
        self._last_reward = reward
        return self._obs(reward, ". ".join(feedback_parts), done=True)

    def _obs(self, reward: float, feedback: str, done: bool) -> DrugDosageObservation:
        t = self._task
        return DrugDosageObservation(
            patient_age=t["patient_age"],
            patient_weight_kg=t["patient_weight_kg"],
            condition=t["condition"],
            allergies=t["allergies"],
            blood_pressure=t["blood_pressure"],
            kidney_function=t["kidney_function"],
            available_drugs=t["available_drugs"],
            feedback=feedback,
            task_name=self._task_name,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state