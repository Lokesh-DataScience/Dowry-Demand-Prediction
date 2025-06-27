from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    model_manager.load_models()
    yield
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_DIR = Path("models")
FEATURE_COLUMNS = [
    'year', "women's age", "men's age", 
    'mohor_log', 'family type_Lower Class', 'family type_Poverty Level', 
    'family type_Rich Class', 'area_Barguna ', 'area_Barishal ', 
    'area_Bhola ', 'area_Brahmanbaria ', 'area_Chandpur ', 
    'area_Chattogram ', "area_Cox’s Bazar ", 'area_Cumilla ', 
    'area_Dhaka ', 'area_Faridpur ', 'area_Gazipur ', 'area_Gopalganj ', 
    'area_Jamalpur ', 'area_Jashore ', 'area_Jhalokati ', 
    'area_Jhenaidah ', 'area_Joypurhat ', 'area_Khagrachhari ', 
    'area_Khulna ', 'area_Kishoreganj ', 'area_Kurigram ', 
    'area_Kushtia ', 'area_Lakshmipur ', 'area_Madaripur ', 
    'area_Magura ', 'area_Manikganj ', 'area_Meherpur ', 
    'area_Munshiganj ', 'area_Narail ', 'area_Narayanganj ', 
    'area_Narsingdi ', 'area_Netrokona ', 'area_Nilphamari ', 
    'area_Panchagarh ', 'area_Patuakhali ', 'area_Pirojpur ', 
    'area_Rajbari ', 'area_Satkhira ', 'area_Shariatpur ', 
    'area_Sherpur ', 'area_Tangail ', 'area_Thakurgaon ', 
    'area_bagerhat ', 'area_bogura ', 'area_chapainawabganj ', 
    'area_chuadanga ', 'area_dinajpur ', 'area_gaibandha ', 
    'area_habiganj ', 'area_lalmonirhat ', 'area_mymensingh ', 
    'area_naogaon ', 'area_natore ', 'area_pabna ', 'area_rajshahi ', 
    'area_rangpur ', 'area_sirajganj ', 'area_sunamganj ', 'area_sylhet ', 
    "girl's job_Day Labour", "girl's job_Service Holder", "girl's job_Worker", 
    "boy's job_Day Labour", "boy's job_Driver", "boy's job_Farmer", 
    "boy's job_Service Holder", "boy's job_Worker",'marry condition_Forced Marriage', 
    'marry condition_Irrelevant Marriage', 'marry condition_Love Marriage',
    'women married/unmarried_Single', 'women married/unmarried_Widow'
]

# Enums for categorical choices
class FamilyType(str, Enum):
    LOWER_CLASS = "Lower Class"
    HIGHER_CLASS = "Higher Class"
    POVERTY_LEVEL = "Poverty Level"
    RICH_CLASS = "Rich Class"

class GirlJob(str, Enum):
    BUSINESS_ENTREPRENEUR = "Business/Entrepreneur" 
    DAY_LABOUR = "Day Labour"
    SERVICE_HOLDER = "Service Holder"
    WORKER = "Worker"

class BoyJob(str, Enum):
    DAY_LABOUR = "Day Labour"
    DRIVER = "Driver"
    FARMER = "Farmer"
    SERVICE_HOLDER = "Service Holder"
    WORKER = "Worker"
    BUSINESS_ENTREPRENEUR = "Business/Entrepreneur"  

class MarryCondition(str, Enum):
    FORCED_MARRIAGE = "Forced Marriage"
    IRRELEVANT_MARRIAGE = "Irrelevant Marriage"
    LOVE_MARRIAGE = "Love Marriage"
    ARRANGE_MARRIAGE = "Arrange Marriage"

class MaritalStatus(str, Enum):
    SINGLE = "Single"
    WIDOW = "Widow"
    DIVORCED = "Divorced"


# Available areas (districts)
AVAILABLE_AREAS = [
    "Bandarban", "Barguna", "Barishal", "Bhola", "Brahmanbaria", "Chandpur", "Chattogram", 
    "Cox's Bazar", "Cumilla", "Dhaka", "Faridpur", "Gazipur", "Gopalganj", 
    "Jamalpur", "Jashore", "Jhalokati", "Jhenaidah", "Joypurhat", "Khagrachhari", 
    "Khulna", "Kishoreganj", "Kurigram", "Kushtia", "Lakshmipur", "Madaripur", 
    "Magura", "Manikganj", "Meherpur", "Munshiganj", "Narail", "Narayanganj", 
    "Narsingdi", "Netrokona", "Nilphamari", "Panchagarh", "Patuakhali", "Pirojpur", 
    "Rajbari", "Satkhira", "Shariatpur", "Sherpur", "Tangail", "Thakurgaon", 
    "Bagerhat", "Bogura", "Chapainawabganj", "Chuadanga", "Dinajpur", "Gaibandha", 
    "Habiganj", "Lalmonirhat", "Mymensingh", "Naogaon", "Natore", "Pabna", 
    "Rajshahi", "Rangpur", "Sirajganj", "Sunamganj", "Sylhet"
]

# Global variables for models
model_manager = None

class ModelManager:
    """Manages model loading and predictions"""
    
    def __init__(self):
        self.rf_model = None
        self.le = None
        self.model_loaded = False
        
    def load_models(self):
        """Load the trained models"""
        try:
            model_path = MODEL_DIR / 'random_forest_model.pkl'
            encoder_path = MODEL_DIR / 'dowry_label_encoder.pkl'
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not encoder_path.exists():
                raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
                
            self.rf_model = joblib.load(model_path)
            self.le = joblib.load(encoder_path)
            self.model_loaded = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load models: {str(e)}"
            )
    
    def predict(self, features_df: pd.DataFrame) -> tuple:
        """Make prediction and return result with confidence"""
        if not self.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Models not loaded"
            )
        
        try:
            # Get prediction
            prediction = self.rf_model.predict(features_df)[0]
            
            # Get prediction probabilities for confidence
            probabilities = self.rf_model.predict_proba(features_df)[0]
            confidence = float(np.max(probabilities))
            
            # Convert prediction to label
            dowry_label = self.le.inverse_transform([prediction])[0]
            
            return dowry_label, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

# Initialize model manager
model_manager = ModelManager()

class SimpleDowryInput(BaseModel):
    """Simplified user-friendly input model"""
    year: int = Field(..., ge=2000, le=2030, description="Year of marriage", example=2023)
    womens_age: float = Field(..., ge=15, le=50, description="Woman's age", example=22.5)
    mens_age: float = Field(..., ge=18, le=60, description="Man's age", example=28.0)
    
    # Financial aspects (users input actual amounts, we'll convert to log)
    mohor_amount: float = Field(..., ge=0, description="Mohor amount in Taka", example=10000.0)
    
    # Categorical selections
    family_type: FamilyType = Field(..., description="Family economic status", example="Poverty Level")
    area: str = Field(..., description="District/Area", example="Dhaka")
    girls_job: GirlJob = Field(GirlJob.BUSINESS_ENTREPRENEUR, description="Woman's occupation", example="Service Holder")
    boys_job: BoyJob = Field(..., description="Man's occupation", example="Service Holder")
    marriage_type: MarryCondition = Field(..., description="Type of marriage", example="Love Marriage")
    womens_marital_status: MaritalStatus = Field(MaritalStatus.SINGLE, description="Woman's marital status", example="Single")
    
    @field_validator('area')
    @classmethod
    def validate_area(cls, v):
        if v not in AVAILABLE_AREAS:
            raise ValueError(f"Area must be one of: {', '.join(AVAILABLE_AREAS)}")
        return v

    @field_validator('mens_age')
    @classmethod
    def validate_age_difference(cls, v, info):
        womens_age = info.data.get('womens_age')
        if womens_age is not None:
            age_diff = abs(v - womens_age)
            if age_diff > 30:
                raise ValueError("Age difference seems unrealistic (>30 years)")
        return v

def convert_to_model_features(input_data: SimpleDowryInput) -> Dict[str, Any]:
    """Convert simplified input to model features with one-hot encoding"""
    
    # Initialize all features to 0
    features = {col: 0 for col in FEATURE_COLUMNS}
    
    # Basic numerical features
    features['year'] = input_data.year
    features["women's age"] = input_data.womens_age
    features["men's age"] = input_data.mens_age
    
    # Convert amounts to log (add small value to avoid log(0))
    features['mohor_log'] = np.log(input_data.mohor_amount + 1)
    
    # One-hot encode family type
    if input_data.family_type == FamilyType.LOWER_CLASS:
        features['family type_Lower Class'] = 1
    elif input_data.family_type == FamilyType.POVERTY_LEVEL:
        features['family type_Poverty Level'] = 1
    elif input_data.family_type == FamilyType.RICH_CLASS:
        features['family type_Rich Class'] = 1
    elif input_data.family_type == FamilyType.HIGHER_CLASS:
        features['family type_Higher Class'] = 1

    # One-hot encode area
    area_column = f"area_{input_data.area} " if input_data.area != "Cox's Bazar" else "area_Cox's Bazar "
    if area_column in features:
        features[area_column] = 1
    else:
        # Handle case variations
        for col in FEATURE_COLUMNS:
            if col.startswith("area_") and input_data.area.lower() in col.lower():
                features[col] = 1
                break
    
    # One-hot encode girl's job
    if input_data.girls_job != GirlJob.BUSINESS_ENTREPRENEUR:  # Business/Entrepreneur is the baseline (all 0s)
        girl_job_col = f"girl's job_{input_data.girls_job.value}"
        if girl_job_col in features:
            features[girl_job_col] = 1
    
    # One-hot encode boy's job
    boy_job_col = f"boy's job_{input_data.boys_job.value}"
    if boy_job_col in features:
        features[boy_job_col] = 1
    
    # One-hot encode marriage condition
    marry_col = f"marry condition_{input_data.marriage_type.value}"
    if marry_col in features:
        features[marry_col] = 1
    
    # One-hot encode marital status
    if input_data.womens_marital_status == MaritalStatus.SINGLE:
        features['women married/unmarried_Single'] = 1
    elif input_data.womens_marital_status == MaritalStatus.WIDOW:
        features['women married/unmarried_Widow'] = 1
    elif input_data.womens_marital_status == MaritalStatus.DIVORCED:
        features['women married/unmarried_Divorced'] = 1

    return features

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_dowry: str
    confidence: float
    timestamp: datetime
    input_summary: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up application...")
    model_manager.load_models()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Dowry Prediction API",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_manager.model_loaded else "unhealthy",
        timestamp=datetime.now(),
        models_loaded=model_manager.model_loaded
    )

@app.get("/areas")
async def get_available_areas():
    """Get list of available areas/districts"""
    return {"areas": sorted(AVAILABLE_AREAS)}

@app.get("/options")
async def get_input_options():
    """Get all available options for categorical fields"""
    return {
        "family_types": [e.value for e in FamilyType],
        "girls_jobs": [e.value for e in GirlJob],
        "boys_jobs": [e.value for e in BoyJob],
        "marriage_types": [e.value for e in MarryCondition],
        "marital_statuses": [e.value for e in MaritalStatus],
        "areas": sorted(AVAILABLE_AREAS)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_dowry(input_data: SimpleDowryInput):
    """
    Predict dowry category based on simplified input
    
    **Easy to use - only provide these fields:**
    - **year**: Year of marriage (2000-2030)
    - **womens_age**: Woman's age (15-50)
    - **mens_age**: Man's age (18-60)
    - **mohor_amount**: Mohor amount in Taka (e.g., 10000)
    - **family_type**: Economic status (Lower Class/Poverty Level/Rich Class)
    - **area**: District name (e.g., Dhaka, Chattogram)
    - **girls_job**: Woman's occupation
    - **boys_job**: Man's occupation
    - **marriage_type**: Type of marriage
    - **womens_marital_status**: Woman's marital status
    """
    try:
        # Convert simplified input to model features
        model_features = convert_to_model_features(input_data)
        
        # Create DataFrame
        df = pd.DataFrame([model_features])
        
        # Ensure all columns are present and in correct order
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        
        # Make prediction
        dowry_label, confidence = model_manager.predict(df)
        
        # Create input summary for response
        input_summary = {
            "year": input_data.year,
            "womens_age": input_data.womens_age,
            "mens_age": input_data.mens_age,
            "age_difference": abs(input_data.mens_age - input_data.womens_age),
            "mohor_amount": input_data.mohor_amount,
            "family_type": input_data.family_type.value,
            "area": input_data.area,
            "marriage_type": input_data.marriage_type.value
        }
        
        return PredictionResponse(
            predicted_dowry=dowry_label,
            confidence=round(confidence, 4),
            timestamp=datetime.now(),
            input_summary=input_summary
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(inputs: List[SimpleDowryInput]):
    """
    Predict dowry categories for multiple inputs (max 50 per request)
    """
    if len(inputs) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 predictions per batch"
        )
    
    results = []
    for i, input_data in enumerate(inputs):
        try:
            result = await predict_dowry(input_data)
            results.append({"index": i, "result": result})
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    
    return {"batch_results": results}

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    try:
        return {
            "model_type": type(model_manager.rf_model).__name__,
            "feature_count": len(FEATURE_COLUMNS),
            "classes": model_manager.le.classes_.tolist(),
            "model_loaded": True,
            "input_fields": {
                "year": "Year of marriage (2000-2030)",
                "womens_age": "Woman's age (15-50)",
                "mens_age": "Man's age (18-60)",
                "mohor_amount": "Mohor amount in Taka",
                "family_type": "Economic status",
                "area": "District/Area",
                "girls_job": "Woman's occupation",
                "boys_job": "Man's occupation",
                "marriage_type": "Type of marriage",
                "womens_marital_status": "Woman's marital status"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )

# Custom exception handler
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )
