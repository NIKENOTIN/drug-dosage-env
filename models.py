from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional

class DrugDosageAction(Action):
    """Action taken by the AI agent — choose a drug and dosage."""
    drug_name: str = Field(..., description="Name of the drug to administer")
    dosage_mg: float = Field(..., description="Dosage in milligrams")
    route: str = Field(default="oral", description="Route: oral, IV, IM")

class DrugDosageObservation(Observation):
    """What the AI agent sees — patient info and current status."""
    patient_age: int = Field(default=0, description="Patient age in years")
    patient_weight_kg: float = Field(default=0.0, description="Patient weight in kg")
    condition: str = Field(default="", description="Medical condition to treat")
    allergies: list[str] = Field(default=[], description="Known drug allergies")
    blood_pressure: str = Field(default="", description="Blood pressure reading")
    kidney_function: str = Field(default="normal", description="normal / impaired / severe")
    available_drugs: list[str] = Field(default=[], description="Drugs available to choose from")
    feedback: str = Field(default="", description="Feedback on last action taken")
    task_name: str = Field(default="", description="Current task name")