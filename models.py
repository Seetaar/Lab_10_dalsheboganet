"""
Pydantic модели для валидации входных данных
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ClientData(BaseModel):
    """Данные клиента для предсказания ипотеки"""

    person_age: float = Field(..., description="Возраст клиента", ge=18, le=100)
    person_income: float = Field(..., description="Годовой доход", ge=0)
    person_emp_exp: float = Field(..., description="Стаж работы (лет)", ge=0)
    loan_amnt: float = Field(..., description="Сумма кредита", ge=0)
    loan_int_rate: float = Field(..., description="Процентная ставка", ge=0, le=100)
    loan_percent_income: float = Field(..., description="Отношение кредита к доходу", ge=0, le=1)
    cb_person_cred_hist_length: float = Field(..., description="Длина кредитной истории (лет)", ge=0)
    credit_score: int = Field(..., description="Кредитный рейтинг", ge=300, le=850)

    person_gender: str = Field(..., description="Пол (male/female)")
    person_education: str = Field(..., description="Образование (Bachelor, Master, High School, Associate, Doctorate)")
    person_home_ownership: str = Field(..., description="Тип владения жильем (RENT, OWN, MORTGAGE, OTHER)")
    loan_intent: str = Field(..., description="Цель кредита (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)")
    previous_loan_defaults_on_file: str = Field(..., description="Были ли дефолты (Yes/No)")

    class Config:
        schema_extra = {
            "example": {
                "person_age": 40,
                "person_income": 2000000,
                "person_emp_exp": 15,
                "loan_amnt": 2000000,
                "loan_int_rate": 8.0,
                "loan_percent_income": 0.15,
                "cb_person_cred_hist_length": 20,
                "credit_score": 800,
                "person_gender": "male",
                "person_education": "Master",
                "person_home_ownership": "OWN",
                "loan_intent": "HOMEIMPROVEMENT",
                "previous_loan_defaults_on_file": "No"
            }
        }


class PredictionResponse(BaseModel):
    loan_status: int = Field(..., description="0 - отказ, 1 - одобрено")
    loan_status_text: str = Field(..., description="Текстовое описание статуса")
    confidence: Optional[float] = Field(None, description="Вероятность предсказания от 0 до 1")
    input_data: Dict[str, Any] = Field(..., description="Исходные данные клиента")


class CSVPredictionResponse(BaseModel):
    message: str
    rows_processed: int
    has_target: bool
    roc_auc: Optional[float] = None
    predictions_file: Optional[str] = None
    sample: Optional[List[Dict[str, Any]]] = None
