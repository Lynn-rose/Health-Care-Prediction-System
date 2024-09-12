import pickle
from typing import Optional
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException, Form, Query, Request, Body
from pydantic import BaseModel
from datetime import date

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/images", StaticFiles(directory="app/static/images"), name="images")
templates = Jinja2Templates(directory="app/templates")


with open('./data/los_model.pkl', 'rb') as file:
    los_model = pickle.load(file)
with open('./data/death_model.pkl', 'rb') as file:
    death_model = pickle.load(file)
with open('./data/readmission_model.pkl', 'rb') as file:
    readmission_model = pickle.load(file)
with open('./data/preprocessor.pkl', 'rb') as file:
    los_preprocessor = pickle.load(file)
with open('./data/death_preprocessor.pkl', 'rb') as file:
    death_preprocessor = pickle.load(file)
with open('./data/readmission_preprocessor.pkl', 'rb') as file:
    readmission_preprocessor = pickle.load(file)

# Load the CSV files
patients = pd.read_csv("app/data/patients.csv")
beds = pd.read_csv("app/data/beds.csv")
staffs = pd.read_csv("app/data/staff.csv")
admissions = pd.read_csv("app/data/admissions.csv")
prescriptions = pd.read_csv("app/data/prescriptions.csv")
diagnoses = pd.read_csv("app/data/diagnoses.csv")
omr = pd.read_csv("app/data/omr.csv")
patients_df = pd.read_csv("app/data/patient_db.csv")
inventory = pd.read_csv("app/data/hospital_inventory.csv")

with open('app/data/marital_statuses.pkl', 'rb') as file:
    marital_statuses = pickle.load(file)
with open('app/data/admission_locations.pkl', 'rb') as file:
    admission_locations = pickle.load(file)
with open('app/data/admission_types.pkl', 'rb') as file:
    admission_types = pickle.load(file)
with open('app/data/races.pkl', 'rb') as file:
    races = pickle.load(file)
with open('app/data/insurance_types.pkl', 'rb') as file:
    insurance_types = pickle.load(file)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/patient_post")
async def api_demo(request: Request):
    
    return templates.TemplateResponse("patient_post.html", context={
        "request": request,
        "marital_statuses": marital_statuses,
        "admission_locations": admission_locations,
        "admission_types": admission_types,
        "races": races,
        "insurance_types": insurance_types,
        "genders": {
            "M": "Male",
            "F": "Female"
        },
    })

@app.post("/predict")
def predict(
    age: int = Form(...),
    prescription: str = Form(...),
    diagnosis: str = Form(...),
    gender: str = Form(...),
    admission_type: str = Form(...),
    admission_location: str = Form(...),
    insurance: str = Form(...),
    marital_status: str = Form(...),
    race: str = Form(...),
    weight: float = Form(...),
    bp_systolic: int = Form(...),
    bp_diastolic: int = Form(...)
):
    try:
        patient = {
            "drug": prescription,
            "age": age,
            "diagnosis": diagnosis,
            "gender": gender,
            "admission_type": admission_type,
            "admission_location": admission_location,
            "insurance": insurance,
            "marital_status": marital_status,
            "race": race,
            "weight": weight,
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic
        }

        results = {}
        df = pd.DataFrame([patient])
        results["los"] = (round(predict_los(df).item()))
        results["death"] = (predict_death_by_data(df).astype(bool).item())
        results["readmission"] = (predict_readmission_by_data(df).astype(bool).item())
        print(results)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))

@app.get("/dashboard")
async def dashboard(request: Request):
    free_beds = beds[beds.adm_id.isna()].shape[0]
    doctors = staffs[(staffs.role == "Physician")].shape[0]
    nurses = staffs[(staffs.role == "Nurse")].shape[0]
    n_patients = admissions[admissions.adm_id.isin(beds.adm_id)].shape[0]
    get_all_patient_predictions()
    patient_count_predictions = predict_patient_counts(n_patients, 2)
    print(patient_count_predictions)

    return templates.TemplateResponse("dashboard.html", context={
        "request": request,
        "beds": beds.to_dict(orient="records"),
        "staffs": staffs.to_dict(orient="records"),
        "patients": patients.to_dict(orient="records"),
        "free_beds": free_beds,
        "doctors": doctors,
        "nurses": nurses,
        "n_patients": n_patients,
        "death_risks": patients[patients.death == True].shape[0],
        "readmission_risks": patients[patients.readmission == True].shape[0],
        "mean_los": patients.los.mean(),
        "max_los": patients.los.max(),
        "patients_tomorrow": patient_count_predictions[0],
        "patients_2days": patient_count_predictions[1],
        "patients_tomorrow_percentage": patient_count_predictions[0] / n_patients * 100,
        "patients_2days_percentage": patient_count_predictions[1] / n_patients * 100,
    })

@app.get("/get_patient_data")
async def get_patient_data(patient_id: int):
    # Search for the patient by ID
    patient_data = patients_df[patients_df['patient_id'] == patient_id]
    
    if patient_data.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Convert the patient data to a dictionary
    patient_data = patient_data.to_dict(orient="records")[0]
    
    return JSONResponse(content=patient_data)

def predict_patient_counts(current, days):
    patient_count = current
    #next day
    stays = patients.los.copy()
    counts = []

    for _ in range(days):
        stays -= 1
        #count instances of 0 in stays
        discharges =  stays[stays == 0].shape[0]
        next_day_patients = patient_count - discharges
        counts.append(next_day_patients)
        patient_count = next_day_patients
    return counts

@app.get("/patients")
async def view_patients(request: Request):
    patients_temp = patients.to_dict(orient="records")
    get_all_patient_predictions()
    return templates.TemplateResponse("patients.html", context={
        "request": request, "patients": patients_temp
    })

@app.get("/staff")
async def staff(request: Request):
    staff_temp = staffs.to_dict(orient="records")
    get_all_staff()
    doctors = staffs[(staffs.role == "Physician")].shape[0]
    nurses = staffs[(staffs.role == "Nurse")].shape[0]
    admin = staffs[(staffs.role == "Admin")].shape[0]
    
    return templates.TemplateResponse("staff.html", context={
        "request": request,
        "staffs": staff_temp,
        "doctors": doctors,
        "nurses": nurses,
        "admin": admin,
    })

@app.post("/match")
async def match():
    try:
        patient = {
             "id": patients[patients.patient_id ],
            "name": patients[patients.name ],
        }
        staff = {
             "id": staffs[staffs.staff_id ],
            "name": staffs[staffs.name ],
            
        }

        results = {}
        patient_df = pd.DataFrame([patient])
        staff_df = pd.DataFrame([staff])
        results["assigned_patients"] = match_doctors_nurses_to_patients(patient_df, staff_df).item()
        print(results)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))
    
@app.get("/inventory")
async def invent(request: Request):
    medical_equipment = inventory[(inventory.category == "Medical Equipment")].shape[0]
    pharmaceuticals = inventory[(inventory.category == "Pharmaceuticals")].shape[0]
    surgical_tools = inventory[(inventory.category == "Surgical Tools")].shape[0]
    ppe = inventory[(inventory.category == "PPE")].shape[0]
    cleaning_supplies = inventory[(inventory.category == "Cleaning Supplies")].shape[0]
    diagnostic_tools = inventory[(inventory.category == "Diagnostic Tools")].shape[0]
    inventory_data = inventory.to_dict(orient="records")  # Ensuring this is a list of dicts
    today = date.today().strftime('%Y-%m-%d')  # Get today's date in 'YYYY-MM-DD' format

    return templates.TemplateResponse("inventory.html", context={
        "request": request,
        "medical_equipment": medical_equipment,
        "pharmaceuticals": pharmaceuticals,
        "surgical_tools": surgical_tools,
        "ppe": ppe,
        "cleaning_supplies": cleaning_supplies,
        "diagnostic_tools": diagnostic_tools,
        "inventory_data": inventory_data,  # Pass inventory data correctly
        "today": today,
    })


@app.get("/inventory_data", response_class=JSONResponse)
async def get_inventory_data():
    # Convert DataFrame to dictionary
    data = inventory.to_dict(orient='records')
    return JSONResponse(content=data)

# Path to the templates folder
#templates = Jinja2Templates(directory="app/templates")

@app.get("/product_details")
async def get_product_details(product_id: int):

    # Find the product by product_id
    product = inventory[inventory["product_id"] == product_id]

    if product.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    # Convert the product details to a dictionary (Pandas DataFrame -> Dict)
    product_details = product.to_dict(orient="records")[0]

    return JSONResponse(content=product_details)

@app.post("/modify_product")
async def modify_product(request: Request):
    data = await request.json()
    product_id = data.get('product_id')
    action = data.get('action')
    quantity = data.get('quantity', 0)
    
    global inventory
    
    # Find the product by product_id
    product_index = inventory[inventory["product_id"] == product_id].index

    if product_index.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    if action == "add":
        # Add the specified quantity to the existing product
        new_quantity = inventory.loc[product_index, "quantity"].values[0] + quantity
        inventory.loc[product_index, "quantity"] = new_quantity

    elif action == "delete":
        # Ensure sufficient quantity before deletion
        current_quantity = inventory.loc[product_index, "quantity"].values[0]
        if current_quantity < quantity:
            raise HTTPException(status_code=400, detail="Not enough stock to delete.")
        new_quantity = current_quantity - quantity
        if new_quantity == 0:
            inventory = inventory.drop(product_index)
        else:
            inventory.loc[product_index, "quantity"] = new_quantity

    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'add' or 'delete'.")

    # Save the updated inventory
    save_inventory(inventory)

    return {"message": f"Product {action} operation successful."}

def save_inventory(df):
    df.to_csv("app/data/hospital_inventory.csv", index=False)

###################################
#Length of stay calculation
####################################
def los_df(patient_id):
    patient = patients[patients.patient_id == patient_id].iloc[0]
    patient_data = {}

    if(patient_id not in admissions.patient_id.values):
        return pd.DataFrame()
    
    adm = admissions[admissions.patient_id == patient_id]
    #print(adm)
    
    adm = adm.iloc[0]['adm_id']
    patient_prescrition = prescriptions[prescriptions.adm_id == adm].iloc[0]
    patient_data['drug'] = patient_prescrition['drug']
    patient_data['age'] = patient['age']
    patient_data['diagnosis'] = diagnoses[diagnoses.adm_id == adm].iloc[0]["diagnosis"]
    patient_data['gender'] = patient['gender']
    patient_data['admission_type'] = admissions[admissions.patient_id == patient_id].iloc[0]['type']
    patient_data["admission_location"] = admissions[admissions.patient_id == patient_id].iloc[0]["location"]
    patient_data["insurance"] = patient['insurance']
    patient_data["marital_status"] = patient['maritalStatus']
    patient_data["race"] = patient['race']
    patient_data['weight'] = omr[omr.adm_id == adm].iloc[0]["weight"]
    patient_data["bp_systolic"] = omr[omr.adm_id == adm].iloc[0]["bp_systolic"]
    patient_data["bp_diastolic"] = omr[omr.adm_id == adm].iloc[0]["bp_diastolic"]

    return pd.DataFrame([patient_data], columns=patient_data.keys())

# Matching function
def match_doctors_nurses_to_patients(patients, staffs):
    doctors = staffs[staffs["role"] == "Physician"]
    nurses = staffs[staffs["role"] == "Nurse"]
    
    doctor_list = doctors[["staff_id", "staff_name"]].to_dict(orient="records")
    nurse_list = nurses[["staff_id", "staff_name"]].to_dict(orient="records")
    
    if len(doctor_list) == 0 or len(nurse_list) == 0:
        raise HTTPException(status_code=400, detail="Not enough doctors or nurses available for matching")
    
    matched_assignments = []
    num_doctors = len(doctor_list)
    num_nurses = len(nurse_list)
    
    for i, patient in patients.iterrows():
        assigned_doctor = doctor_list[i % num_doctors]
        assigned_nurse = nurse_list[i % num_nurses]
        
        matched_assignments.append({
            "patient_id": patient["patient_id"],
            "patient_name": patient["patient_name"],
            "assigned_doctor": assigned_doctor["staff_name"],
            "assigned_nurse": assigned_nurse["staff_name"]
        })
    
    return matched_assignments


def get_all_patient_predictions():
    predictions = {
        "los": [],
        "death": [],
        "readmission": []
    }
    drugs = []
    diag = []
    for i in range(len(patients)):
        patient_id = patients.iloc[i]['patient_id']
        df = los_df(patient_id)
        drugs.append(df.iloc[0].drug)
        diag.append(df.iloc[0].diagnosis)
        #print(df.columns)
        predictions["los"].append(predict_los(df).astype(int))
        predictions["death"].append(predict_death(patient_id).astype(bool))
        predictions["readmission"].append(predict_readmission(patient_id).astype(bool))
    patients['los'] = predictions["los"]
    patients['death'] = predictions['death']
    patients['readmission'] = predictions['readmission']
    patients["drug"] = drugs
    patients["diagnosis"] = diag
    print(patients.iloc[0])

def get_all_staff():
    name = []
    shift_start = []
    shift_end = []
    for i in range(len(staffs)):
        staff_id = staffs.iloc[i]['staff_id']
        name.append(staffs.iloc[i].staff_name)
        shift_start.append(staffs.iloc[i].shift_start)
        shift_end.append(staffs.iloc[i].shift_end)
        #print(df.columns)
        
    staffs["staff_name"] = name
    staffs["shift_start"] = shift_start
    staffs["shift_end"] = shift_end
    print(staffs.iloc[i])

def predict_los(patient_data):
    patient_data_transformed = los_preprocessor.transform(patient_data)
    return los_model.predict(patient_data_transformed)[0]
def predict_death_by_data(data):
    data_transformed = death_preprocessor.transform(data)
    return death_model.predict(data_transformed)[0]
def predict_readmission_by_data(data):
    data_transformed = readmission_preprocessor.transform(data)
    return readmission_model.predict(data_transformed)[0]

def predict_death(patient_id):
    data = los_df(patient_id)
    data_transformed = death_preprocessor.transform(data)
    return death_model.predict(data_transformed)[0]

def predict_readmission(patient_id):
    data = los_df(patient_id)
    data_transformed = readmission_preprocessor.transform(data)
    return readmission_model.predict(data_transformed)[0]


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
