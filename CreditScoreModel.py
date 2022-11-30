from pydantic import BaseModel

class CreditScoreModel(BaseModel):
    
        month:int
        age:int
        annual_income:float
        monthly_inhand_salary:float
        num_bank_accounts:int
        num_credit_card:int
        interest_rate:int
        num_of_loan:int
        delay_from_due_date:int
        num_of_delayed_payment:float
        changed_credit_limit:float
        num_credit_inquiries:float
        credit_mix:int
        outstanding_debt:float
        credit_utilization_ratio:float
        credit_history_age:float
        total_emi_per_month:float
        amount_invested_monthly:float
        monthly_balance:float
        accountant:float
        architect:float
        developer:float
        doctor:float
        engineer:float
        entrepreneur:float
        journalist:float
        lawyer:float
        manager:float
        mechanic:float
        media_manager:float
        musician:float
        scientist:float
        teacher:float
        writer:float
        nm:float
        no:float
        yes:float
        high_spent_large_value_payments:float
        high_spent_medium_value_payments:float
        high_spent_small_value_payments:float
        low_spent_large_value_payments:float
        low_spent_medium_value_payments:float
        low_spent_small_value_payments:float