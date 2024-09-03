"""Module to set up patient and staff data"""

import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker
fake = Faker()

# Generate mock patient data
def generate_patients(n):
    patients = []
    for _ in range(n):
        patients.append({
            'Patient ID': fake.unique.uuid4(),
            'Name': fake.name(),
            'Age': np.random.randint(1, 90),
            'Gender': np.random.choice(['Male', 'Female']),
            #'Contact': fake.phone_number(),
            #'Address': fake.address(),
        })
    return pd.DataFrame(patients)

# Generate mock staff data
def generate_staff(n):
    staff = []
    for _ in range(n):
        staff.append({
            'Staff ID': fake.unique.uuid4(),
            'Name': fake.name(),
            'Role': np.random.choice(['Physician', 'Nurse', 'Admin']),
            'Shift Start': fake.time(),
            'Shift End': fake.time()
        })
    return pd.DataFrame(staff)

# Generate mock inventory data
def generate_inventory(n):
    inventory = []
    for _ in range(n):
        inventory.append({
            'Item ID': fake.unique.uuid4(),
            'Item Name': fake.word(),
            'Quantity': np.random.randint(1, 100),
            'Expiry Date': fake.date_between(start_date='today', end_date='+2y')
        })
    return pd.DataFrame(inventory)

# Generate mock prescriptions data
def generate_prescriptions(n, patients, inventory):
    prescriptions = []
    for _ in range(n):
        prescriptions.append({
            'Prescription ID': fake.unique.uuid4(),
            'Patient ID': np.random.choice(patients['Patient ID']),
            'Medication ID': np.random.choice(inventory['Item ID']),
            'Date': fake.date_this_year()
        })
    return pd.DataFrame(prescriptions)

# Generate data
num_patients = 10
num_staff = 5
num_inventory = 20
num_prescriptions = 15

patients_df = generate_patients(num_patients)
staff_df = generate_staff(num_staff)
inventory_df = generate_inventory(num_inventory)
prescriptions_df = generate_prescriptions(num_prescriptions, patients_df, inventory_df)

# Display data
#patients_df.to_csv("app/hospital_data/patients.csv", index=False)
#staff_df.to_csv("app/hospital_data/staff.csv", index=False)
#inventory_df.to_csv("app/hospital_data/inventory.csv", index=False)
#prescriptions_df.to_csv("app/hospital_data/prescriptions.csv", index=False)

with open('app/hospital_data/patients.csv', 'w') as f:
    patients_df.to_csv(f, index=False)
with open('app/hospital_data/staff.csv', 'w') as f:
    staff_df.to_csv(f, index=False)
with open('app/hospital_data/inventory.csv', 'w') as f:
    inventory_df.to_csv(f, index=False)
with open('app/hospital_data/prescriptions.csv', 'w') as f:
    prescriptions_df.to_csv(f, index=False)


