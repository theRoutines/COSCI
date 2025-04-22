from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import json
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import uuid
import base64
import warnings
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from io import BytesIO
from PIL import Image, ImageDraw
import urllib3
import replicate
import time

# Initialize FastAPI app
app = FastAPI(title="COSCI API", version="1.0.0")

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 40)
    logger.info("Server is running!")
    logger.info(f"To access the API documentation, visit: /docs")
    logger.info(f"To access the alternative API documentation, visit: /redoc")
    logger.info("=" * 40)

@app.get("/")
async def serve_spa(request: Request):
    logger.info(f"Accessing root path from {request.client.host}")
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return {"status": "API is running", "message": "Welcome to COSCI API"}

@app.get("/{full_path:path}")
async def serve_spa_paths(full_path: str):
    if not os.path.exists(f"static/{full_path}"):
        return FileResponse("static/index.html")
    return FileResponse(f"static/{full_path}")

# Load environment variables (only in development)
if os.path.exists('.env'):
    load_dotenv()

# Load environment variables with fallbacks
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Log missing variables without raising errors
for var_name in ['GOOGLE_API_KEY', 'HF_API_TOKEN']:
    if not os.getenv(var_name):
        print(f"Warning: {var_name} not found in environment variables")

# Configure services only if API keys are available
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Hugging Face configuration
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# Configure Replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_KEY")
if REPLICATE_API_TOKEN:
    replicate.Client(api_token=REPLICATE_API_TOKEN)
else:
    print("Warning: REPLICATE_API_KEY not found in environment variables")

# Add at the top after imports
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a session with connection pooling
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[500, 502, 503, 504, 429],
    allowed_methods=["POST"],
    raise_on_status=False
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=5,
    pool_maxsize=5,
    pool_block=True
)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./cosplay.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class Character(Base):
    __tablename__ = "characters"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    fandom = Column(String)
    gender_personality = Column(String)
    costume_details = Column(String)
    accessories = Column(String)
    backstory = Column(String)
    catchphrase = Column(String)
    image_url = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    rating = Column(Integer, default=0)  # 0 = not rated, 1 = favorite

Base.metadata.create_all(bind=engine)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://d19hfq71y1mtyx.amplifyapp.com",  # Main Amplify domain
        "https://staging.d19hfq71y1mtyx.amplifyapp.com",  # Staging domain
        "http://localhost:3000"  # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatInput(BaseModel):
    message: str

class ImageInput(BaseModel):
    description: str

class CharacterInput(BaseModel):
    name: str
    fandom: str
    gender_personality: str
    costume_details: str
    accessories: str
    backstory: str
    catchphrase: str
    image_url: str = None
    details: dict = None
    rating: int = 0

def generate_with_huggingface(prompt: str) -> dict:
    """Generate image using Hugging Face's Stable Diffusion API"""
    try:
        print("Starting Hugging Face generation...")
        
        if not HF_API_TOKEN:
            print("Error: Hugging Face API token not found")
            raise ValueError("Hugging Face API token not found in environment variables")

        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        print("Debug - Headers prepared with token")

        # Add some styling parameters to enhance the image
        enhanced_prompt = f"{prompt}, highly detailed, cinematic lighting, 8k, photorealistic, cosplay character, full body shot, intricate details, professional photography"
        negative_prompt = "blurry, low quality, distorted, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, boring, watermark, signature, text, grainy"

        print(f"Using prompt: {enhanced_prompt}")
        
        # Create generation request
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "width": 768,
                "height": 768,
                "num_inference_steps": 50,
                "guidance_scale": 7.5
            }
        }

        print("Calling Hugging Face API...")
        # Make the API request
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Error in Hugging Face API call: {response.text}")
            raise Exception(f"Hugging Face API error: {response.text}")

        # The response is the image bytes
        image_bytes = response.content
        
        # Convert to base64
        img_str = base64.b64encode(image_bytes).decode()
        print("Image successfully generated and converted to base64")
        
        return {
            "image_url": f"data:image/png;base64,{img_str}",
            "description": prompt
        }

    except Exception as e:
        print(f"Error in generate_with_huggingface: {str(e)}")
        raise Exception(f"Error generating image with Hugging Face: {str(e)}")

def generate_with_gemini(prompt: str) -> dict:
    """Generate image description using Google's Gemini"""
    try:
        print("Starting Gemini generation...")
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: Google API key not found")
            raise ValueError("Google API key not found in environment variables")

        # Initialize the model with the correct name
        print("Initializing Gemini model...")
        model = genai.GenerativeModel('gemini-1.5-pro')

        # Generate the image description
        image_prompt = f"""Create a detailed image description for a cosplay character based on this: {prompt}
        The description should be detailed enough to generate a high-quality image.
        Focus on:
        - Character's appearance
        - Costume details
        - Pose and expression
        - Background elements
        - Lighting and atmosphere
        
        Format the response as a clear, detailed image description."""

        print("Generating image description...")
        # Get the image description
        response = model.generate_content(image_prompt)
        image_description = response.text
        print(f"Generated description: {image_description}")

        # Generate the actual image using Hugging Face
        print("Starting image generation with Hugging Face...")
        result = generate_with_huggingface(image_description)
        print("Image generation completed successfully")
        
        return result

    except Exception as e:
        print(f"Error in generate_with_gemini: {str(e)}")
        raise Exception(f"Error generating image with Gemini: {str(e)}")

@app.post("/create-character")
async def create_character(input: ChatInput, db: Session = Depends(get_db)):
    try:
        print(f"Received character creation request: {input.message}")
        
        # Generate character description and image using Gemini
        character_prompt = f"""Create a detailed cosplay character based on this description: {input.message}.
        The character should have:
        - A unique name
        - A specific fandom or universe
        - Gender and personality traits
        - Detailed costume description
        - Special accessories
        - A short backstory
        - A memorable catchphrase
        
        Format the response in this exact structure:
        Name: [character name]
        Fandom/Universe: [fandom or universe]
        Gender & Personality: [gender and personality traits]
        Costume Details: [detailed costume description]
        Accessories: [special accessories]
        Backstory: [short backstory]
        Catchphrase: "[catchphrase]"
        
        Each section should be on a new line and start with the section name followed by a colon."""
        
        result = generate_with_gemini(character_prompt)
        
        # Parse the description into structured sections
        description = result["description"]
        sections = {
            "Name": "",
            "Fandom/Universe": "",
            "Gender & Personality": "",
            "Costume Details": "",
            "Accessories": "",
            "Backstory": "",
            "Catchphrase": ""
        }
        
        # Split the description into lines and parse each section
        for line in description.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a section header
            for section in sections.keys():
                if line.startswith(f"{section}:"):
                    # Extract the content after the colon
                    content = line[len(section) + 1:].strip()
                    sections[section] = content
                    break
        
        # Validate that we have at least some content in each section
        for section, content in sections.items():
            if not content or content.lower() in ["unknown", "no details provided", "no accessories", "no backstory", "no catchphrase"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to generate proper {section.lower()}. Please try again with a different prompt."
                )
        
        # Create a new character in the database with proper values
        db_character = Character(
            name=sections["Name"],
            fandom=sections["Fandom/Universe"],
            gender_personality=sections["Gender & Personality"],
            costume_details=sections["Costume Details"],
            accessories=sections["Accessories"],
            backstory=sections["Backstory"],
            catchphrase=sections["Catchphrase"],
            image_url=result["image_url"]
        )
        
        db.add(db_character)
        db.commit()
        db.refresh(db_character)
        
        # Format the reply with clear sections
        formatted_reply = f"""Created a new character! Here are the details:

Name: {db_character.name}
Fandom/Universe: {db_character.fandom}
Gender & Personality: {db_character.gender_personality}

Costume Details:
{db_character.costume_details}

Accessories:
{db_character.accessories}

Backstory:
{db_character.backstory}

Catchphrase: "{db_character.catchphrase}"
"""
        
        return {
            "reply": formatted_reply,
            "character": {
                "id": db_character.id,
                "name": db_character.name,
                "fandom": db_character.fandom,
                "gender_personality": db_character.gender_personality,
                "costume_details": db_character.costume_details,
                "accessories": db_character.accessories,
                "backstory": db_character.backstory,
                "catchphrase": db_character.catchphrase,
                "image_url": db_character.image_url
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in create-character endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(input: ImageInput):
    try:
        # Validate input description
        if not input.description or len(input.description.strip()) == 0:
            return {
                "error": "Image description cannot be empty",
                "image_url": None,
                "description": None
            }
        
        # Generate image using Gemini
        result = generate_with_gemini(input.description)
        
        return {
            "image_url": result["image_url"],
            "description": result["description"],
            "error": None
        }
        
    except Exception as e:
        error_message = f"Error in generate-image endpoint: {str(e)}"
        print(error_message)
        return {
            "error": error_message,
            "image_url": None,
            "description": None
        }

@app.post("/characters")
async def create_character_manual(character: CharacterInput, db: Session = Depends(get_db)):
    try:
        # Validate that the character has all required fields
        required_fields = {
            "name": character.name,
            "fandom": character.fandom,
            "gender_personality": character.gender_personality,
            "costume_details": character.costume_details,
            "accessories": character.accessories,
            "backstory": character.backstory,
            "catchphrase": character.catchphrase
        }
        
        # Check for empty or default values
        for field, value in required_fields.items():
            if not value or value == "Unknown":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Field '{field}' cannot be empty or 'Unknown'"
                )
        
        # Create the character
        db_character = Character(
            name=character.name,
            fandom=character.fandom,
            gender_personality=character.gender_personality,
            costume_details=character.costume_details,
            accessories=character.accessories,
            backstory=character.backstory,
            catchphrase=character.catchphrase,
            image_url=character.image_url,
            details=character.details,
            rating=character.rating
        )
        
        db.add(db_character)
        db.commit()
        db.refresh(db_character)
        return db_character
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error saving character: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/characters")
async def get_characters(db: Session = Depends(get_db)):
    characters = db.query(Character).all()
    return characters

@app.get("/characters/{character_id}")
async def get_character(character_id: int, db: Session = Depends(get_db)):
    character = db.query(Character).filter(Character.id == character_id).first()
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return character

@app.delete("/characters/{character_id}")
async def delete_character(character_id: int, db: Session = Depends(get_db)):
    character = db.query(Character).filter(Character.id == character_id).first()
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    
    db.delete(character)
    db.commit()
    return {"message": "Character deleted successfully"}

@app.put("/characters/{character_id}/rate")
async def rate_character(character_id: int, rating: int, db: Session = Depends(get_db)):
    character = db.query(Character).filter(Character.id == character_id).first()
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    
    character.rating = rating
    db.commit()
    db.refresh(character)
    return character

@app.get("/test-replicate")
async def test_replicate():
    try:
        print("Testing Replicate API connection...")
        if not REPLICATE_API_TOKEN:
            return {"error": "Replicate API key not found in environment variables"}
        
        # Test the API key with a simple request
        test_output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": "test",
                "width": 64,
                "height": 64
            }
        )
        
        return {"status": "success", "message": "Replicate API connection successful"}
    except Exception as e:
        return {"error": str(e)}

@app.on_event("shutdown")
def shutdown_event():
    session.close()
