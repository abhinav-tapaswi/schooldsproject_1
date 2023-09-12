from joblib import load
import numpy as np
data={
    "Dermatologist":["Fungal infection","Acne","Psoriasis","Impetigo"],
    "Gastroenterologist":["hepatitis A","Hepatitis B","Hepatitis C","Hepatitis D","Hepatitis E","Alcoholic hepatitis","GERD","Chronic cholestasis","Peptic ulcer diseae","Gastroenteritis","Jaundice"],
    "Pulmonologist":["Tuberculosis","Bronchial Asthma"],
    "General Physician":["Common Cold","AIDS","Hypertension","Malaria","Chicken pox","Dengue","Typhoid"],
    "Lung Specialist":["Pneumonia"],
    "Proctologist":["Dimorphic hemmorhoids(piles)"],
    "Cardiologist":["Heart attack"],
    "Vascular Surgeon":["Varicose veins"],
    "Endocrinologist":["Hypothyroidism","Hyperthyroidism","Hypoglycemia","Diabetes"],
    "Rheumatologist":["Osteoarthritis","Arthritis"],
    "ENT Specialist":["(vertigo) Paroymsal  Positional Vertigo"],
    "Urologist":["Urinary Tract Infection"],
    "Allergist":["Allergy","Drug Reaction"],
    "Neurologist":["Migraine","Paralysis (brain hemorrhage)"],
    "Orthopaedic Surgeon":["Cervical spondylosis"]
}
class DiseasetoDoctor:
    def findDoc(self,lis):
        model=load('symptoms-disease_model.joblib')
        pred=model.predict(np.array([lis]))
        for i in pred:
            p=i
        for key, value_list in data.items():
            if p in value_list:
                found_key = key
                break
        return [pred,found_key]
