"""Request and response schemas for the inference API."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Job(str, Enum):
    """Supported customer occupations."""

    ADMIN = "admin."
    BLUE_COLLAR = "blue-collar"
    ENTREPRENEUR = "entrepreneur"
    HOUSEMAID = "housemaid"
    MANAGEMENT = "management"
    RETIRED = "retired"
    SELF_EMPLOYED = "self-employed"
    SERVICES = "services"
    STUDENT = "student"
    TECHNICIAN = "technician"
    UNEMPLOYED = "unemployed"
    UNKNOWN = "unknown"


class MaritalStatus(str, Enum):
    """Supported marital statuses."""

    DIVORCED = "divorced"
    MARRIED = "married"
    SINGLE = "single"


class Education(str, Enum):
    """Supported education levels."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    UNKNOWN = "unknown"


class ContactType(str, Enum):
    """Supported contact channels."""

    CELLULAR = "cellular"
    TELEPHONE = "telephone"
    UNKNOWN = "unknown"


class Month(str, Enum):
    """Supported abbreviated month values."""

    JAN = "jan"
    FEB = "feb"
    MAR = "mar"
    APR = "apr"
    MAY = "may"
    JUN = "jun"
    JUL = "jul"
    AUG = "aug"
    SEP = "sep"
    OCT = "oct"
    NOV = "nov"
    DEC = "dec"


class PreviousOutcome(str, Enum):
    """Supported outcomes from a previous campaign."""

    FAILURE = "failure"
    OTHER = "other"
    SUCCESS = "success"
    UNKNOWN = "unknown"


class PredictionRequest(BaseModel):
    """Raw customer and current campaign values required by the model."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "age": 42,
                    "job": "management",
                    "marital": "married",
                    "education": "tertiary",
                    "default": "no",
                    "balance": 1850,
                    "housing": "yes",
                    "loan": "no",
                    "contact": "cellular",
                    "day": 15,
                    "month": "may",
                    "duration": 320,
                    "campaign": 2,
                    "pdays": -1,
                    "previous": 0,
                    "poutcome": "unknown",
                }
            ]
        },
    )

    age: int = Field(ge=18, le=120)
    job: Job
    marital: MaritalStatus
    education: Education
    default: Literal["no", "yes"]
    balance: int
    housing: Literal["no", "yes"]
    loan: Literal["no", "yes"]
    contact: ContactType
    day: int = Field(ge=1, le=31)
    month: Month
    duration: int = Field(
        ge=1,
        description=(
            "Current call duration in seconds. This value is only known "
            "after the call and limits pre-contact use of the model."
        ),
    )
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: PreviousOutcome

    @field_validator(
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
        mode="before",
    )
    @classmethod
    def normalize_categorical_value(cls, value: object) -> object:
        """Apply the deterministic categorical normalization used in training."""
        if isinstance(value, str):
            return value.strip().lower()
        return value


class PredictionResponse(BaseModel):
    """Model output for one customer."""

    prediction: Literal[0, 1]
    subscription_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Inference service readiness response."""

    status: Literal["healthy"]
    model_loaded: Literal[True]
    model_artifact: str
