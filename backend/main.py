import shutil
import os
import re
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
from PIL import Image
import pytesseract

app = FastAPI(title="Insurance Fraud Detection API", version="1.0.0")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],   
)

# -------------------- DATASET LOADERS --------------------

def load_hospitals():
    """Loads the hospital dataset and cleans up the column names."""
    file_path = os.path.join(os.path.dirname(__file__), "datasets", "Hospital_Dataset.xlsx")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hospital dataset not found at {file_path}")
    hospitals_df = pd.read_excel(file_path)
    hospitals_df.columns = hospitals_df.columns.str.strip()
    return hospitals_df

def load_diseases():
    """Loads the disease-treatment dataset and cleans up the column names."""
    file_path = os.path.join(os.path.dirname(__file__), "datasets", "disease_treatment_dataset.xlsx")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Disease dataset not found at {file_path}")
    diseases_df = pd.read_excel(file_path)
    diseases_df.columns = diseases_df.columns.str.strip()
    return diseases_df

# -------------------- FRAUD CHECKING --------------------

# -------------------- FRAUD CHECKING --------------------

def check_fraud(claim_data, hospitals_df, diseases_df):
    """
    Checks for fraud based on:
    1. Hospital name + region + pincode match
    2. Disease validation
    3. Treatment validation for the disease
    4. Claim amount validation (suspicious if too high)
    """
    hospital_name = claim_data.get("hospital", "").strip()
    region = claim_data.get("region", "").strip()
    pincode = claim_data.get("pincode", "").strip()
    disease = claim_data.get("disease", "").strip()
    treatment = claim_data.get("treatment", "").strip()
    amount = claim_data.get("amount", 0)

    print("\n--- Checking for Fraud ---")
    print(f"Hospital: {hospital_name}, Region: {region}, Pincode: {pincode}")
    print(f"Disease: {disease}, Treatment: {treatment}, Amount: {amount}")

    # ✅ Check 1: Hospital must match Name, Region, and Pincode
    if hospital_name and region and pincode:
        hospital_row = hospitals_df[
            (hospitals_df['HospitalName'].str.lower() == hospital_name.lower()) &
            (hospitals_df['Region'].str.lower() == region.lower()) &
            (hospitals_df['Pincode'].astype(str) == str(pincode))
        ]
        if hospital_row.empty:
            return "fraudulent", "Missing hospital, region, or pincode information"
    else: 
        return "fraudulent", "Hospital does not match given region or pincode"

    # ✅ Check 2: Disease and treatment validation
    if disease and treatment:
        # Find the disease row in the dataset
        disease_row = diseases_df[diseases_df['Disease'].str.lower() == disease.lower()]
        
        if disease_row.empty:
            return "fraudulent", f"Disease '{disease}' not found in dataset"
        
        # Get all valid treatments for this disease
        valid_treatments = [t.strip().lower() for t in disease_row.iloc[0]['Treatment'].split(',')]
        
        # Check if the provided treatment matches exactly
        if treatment.lower() not in valid_treatments:
            return "fraudulent", f"Treatment '{treatment}' does not match disease '{disease}'"
    else:
        return "fraudulent", "Missing disease or treatment information"

    # ✅ Check 3: Claim amount validation
    if hospital_row is not None and amount > 0:
        avg_amount = hospital_row.iloc[0]['AvgTreatmentCost']
        # Flag as suspicious if claim exceeds 1.5x average
        if amount > avg_amount * 1.5:
            return "suspicious", f"Claim amount ₹{amount} unusually high compared to average ₹{avg_amount}"

    # ✅ If all checks pass
    return "clean", "All details verified successfully"

# -------------------- FILE TEXT EXTRACTION --------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(file_path: str) -> str:
    """Extracts text from an image using pytesseract OCR."""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# -------------------- LOAD DATASETS --------------------
try:
    hospitals_df = load_hospitals()
    diseases_df = load_diseases()
    print("Datasets loaded successfully")
except Exception as e:
    print(f"Error loading datasets: {e}")
    hospitals_df = pd.DataFrame({'HospitalName': [], 'Region': [], 'Pincode': []})
    diseases_df = pd.DataFrame({'Disease': [], 'Treatment': []})

# -------------------- ROUTES --------------------

@app.get("/")
async def root():
    return {"message": "Insurance Fraud Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "hospitals_count": len(hospitals_df), "diseases_count": len(diseases_df)}

@app.post("/api/claims/upload")
async def upload_claim(file: UploadFile = File(...)):
    """Handles file upload, extracts text, and performs a fraud check."""
    temp_file_path = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text
        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(temp_file_path)
        else:
            text = extract_text_from_image(temp_file_path)

        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from document.")

        print(f"Extracted text preview: {text[:500]}...")

        # Extract data using regex
        hospital_match = re.search(r"Hospital(?:\s+Name)?:?\s*(.+)", text, re.IGNORECASE)
        region_match = re.search(r"Region:?\s*(.+)", text, re.IGNORECASE)
        pincode_match = re.search(r"Pincode:?\s*(\d+)", text, re.IGNORECASE)
        disease_match = re.search(r"Disease[:\-\s]+(.+)", text, re.IGNORECASE)
        treatment_match = re.search(r"Treatment[:\-\s]+(.+)", text, re.IGNORECASE)
        amount_match = re.search(r"(?:Claimed\s+)?Amount:?\s*[₹$]?\s*([\d,\.]+)", text, re.IGNORECASE)
        patient_name_match = re.search(r"(?:Policy\s+Holder|Patient)\s+Name:?\s*(.+)", text, re.IGNORECASE)
        claim_id_match = re.search(r"Claim\s+(?:No|ID|Number):?\s*([\w\d/]+)", text, re.IGNORECASE)

        hospital = hospital_match.group(1).strip() if hospital_match else ""
        region = region_match.group(1).strip() if region_match else ""
        pincode = pincode_match.group(1).strip() if pincode_match else ""
        disease = disease_match.group(1).strip() if disease_match else ""
        treatment = treatment_match.group(1).strip() if treatment_match else ""
        amount = int(float(amount_match.group(1).replace(",", ""))) if amount_match else 0
        patient_name = patient_name_match.group(1).strip() if patient_name_match else ""
        claim_id = claim_id_match.group(1).strip() if claim_id_match else ""

        claim_data = {
            "hospital": hospital,
            "region": region,
            "pincode": pincode,
            "disease": disease,
            "treatment": treatment,
            "amount": amount,
            "patientName": patient_name,
            "claimId": claim_id,
        }

        print("\n--- Extracted Data ---")
        print(claim_data)

        # Perform fraud check
        fraud_status, fraud_reason = check_fraud(claim_data, hospitals_df, diseases_df)
        claim_data["fraudStatus"] = fraud_status
        claim_data["fraudReason"] = fraud_reason

        return JSONResponse(content={"status": "success", "extractedData": claim_data}, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
