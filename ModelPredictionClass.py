from joblib import load
import numpy as np
import random
import pandas as pd

class PredictionModel:

    def __init__(self):
        self.model = load("final_model/finalized_model.sav")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result

    def generate_random_data(self, size, dataIndex = False):
        data = []
        df = pd.DataFrame(columns=["month","age","annual_income","monthly_inhand_salary","num_bank_accounts","num_credit_card","interest_rate","num_of_loan","delay_from_due_date","num_of_delayed_payment","changed_credit_limit","num_credit_inquiries","credit_mix","outstanding_debt","credit_utilization_ratio","credit_history_age","total_emi_per_month","amount_invested_monthly","monthly_balance","accountant","architect","developer","doctor","engineer","entrepreneur","journalist","lawyer","manager","mechanic","media_manager","musician","scientist","teacher","writer","nm","no","yes","high_spent_large_value_payments","high_spent_medium_value_payments","high_spent_small_value_payments","low_spent_large_value_payments","low_spent_medium_value_payments","low_spent_small_value_payments"])

        professions = ["accountant","architect","developer","doctor","engineer","entrepreneur","journalist","lawyer","manager","mechanic","media_manager","musician","scientist","teacher","writer"]
        profession = np.random.choice(professions, size=1)[0]
        
        min_amounts = ["nm","no","yes"]
        min_amount = np.random.choice(min_amounts, size=1)[0]

        behaviors = ["high_spent_large_value_payments","high_spent_medium_value_payments","high_spent_small_value_payments","low_spent_large_value_payments","low_spent_medium_value_payments","low_spent_small_value_payments"]
        behavior = np.random.choice(behaviors, size=1)[0]
        
        for i in range(0,size):
            new_data = {
            "month":random.randint(1,12),
                "age":random.randint(14,60),
                "annual_income":random.randint(65000,1.4E+10),
                "monthly_inhand_salary":random.randint(65000,1.4E+10),
                "num_bank_accounts":random.randint(0,12),
                "num_credit_card":random.randint(0,12),
                "interest_rate":random.randint(1,34),
                "num_of_loan":random.randint(0,9),
                "delay_from_due_date":random.randint(0,70),
                "num_of_delayed_payment":random.randint(0,300),
                "changed_credit_limit":random.randint(0,1.4E+10),
                "num_credit_inquiries":random.randint(0,200),
                "credit_mix":random.randint(1,3),
                "outstanding_debt":random.randint(10,500000),
                "credit_utilization_ratio":random.randint(2.5E+12,5E+16),
                "credit_history_age":random.randint(10,4000),
                "total_emi_per_month":random.randint(9E+10,8E+16),
                "amount_invested_monthly":random.randint(1.5E+12,6.5E+16),
                "monthly_balance":random.randint(1.8E+12,6.E+16),
                "accountant":1 if profession == "accountant" else 0,
                "architect":1 if profession == "architect" else 0,
                "developer":1 if profession == "developer" else 0,
                "doctor":1 if profession == "doctor" else 0,
                "engineer":1 if profession == "engineer" else 0,
                "entrepreneur":1 if profession == "entrepreneur" else 0,
                "journalist":1 if profession == "journalist" else 0,
                "lawyer":1 if profession == "lawyer" else 0,
                "manager":1 if profession == "manager" else 0,
                "mechanic":1 if profession == "mechanic" else 0,
                "media_manager":1 if profession == "media_manager" else 0,
                "musician":1 if profession == "musician" else 0,
                "scientist":1 if profession == "scientist" else 0,
                "teacher":1 if profession == "teacher" else 0,
                "writer":1 if profession == "writer" else 0,
                "nm":1 if min_amount == "nm" else 0,
                "no":1 if min_amount == "no" else 0,
                "yes":1 if min_amount == "yes" else 0,
                "high_spent_large_value_payments":1 if behavior == "high_spent_large_value_payments" else 0,
                "high_spent_medium_value_payments":1 if behavior == "high_spent_medium_value_payments" else 0,
                "high_spent_small_value_payments":1 if behavior == "high_spent_small_value_payments" else 0,
                "low_spent_large_value_payments":1 if behavior == "low_spent_large_value_payments" else 0,
                "low_spent_medium_value_payments":1 if behavior == "low_spent_medium_value_payments" else 0,
                "low_spent_small_value_payments":1 if behavior == "low_spent_small_value_payments" else 0
            }

            if(dataIndex==True):
                data.append(new_data)
            
            df = df.append(new_data, ignore_index=True)
        
        if(dataIndex==True):
            print(data)

        return df

#prediction = PredictionModel().make_predictions(PredictionModel().generate_random_data(1))
#print(prediction)

PredictionModel().generate_random_data(10, dataIndex=True)