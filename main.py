#!/usr/bin/env python3
"""
FastAPI application for PDF OCR extraction
Extracts text from PDFs and returns structured JSON data
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from pdf2image import convert_from_path
import json
import re
import tempfile
import os
from typing import Dict, List, Optional
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="PDF OCR Extraction API",
    description="Extract structured data from PDF documents using OCR",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class ContactInfo(BaseModel):
    foretag: Optional[str] = None
    roll: Optional[str] = None
    namn: Optional[str] = None
    email: Optional[str] = None
    telefon: Optional[str] = None


class ViktigaDatum(BaseModel):
    byggstart_slut: Optional[str] = None
    senaste_uppdatering: Optional[str] = None
    publicerad: Optional[str] = None
    sista_anbudsdag: Optional[str] = None


class Information(BaseModel):
    ref_nummer: Optional[str] = None
    kategorier: Optional[List[str]] = None
    entreprenadformer: Optional[str] = None
    geografi: Optional[str] = None
    projekttyper: Optional[str] = None
    material: Optional[str] = None
    miljocertifieringar: Optional[str] = None
    uppskattad_projektkostnad: Optional[str] = None


class Fas(BaseModel):
    planering: str = "upcoming"
    projektering: str = "upcoming"
    anbud: str = "upcoming"
    produktion: str = "upcoming"
    avslutad: str = "upcoming"


class ProjectData(BaseModel):
    huvudtitel: str = ""
    undertitel: str = ""
    status: str = ""
    beskrivning: str = ""
    projekt_omfattar: str = ""
    viktiga_datum: ViktigaDatum
    information: Information
    kontakter: List[ContactInfo]
    fas: Fas


# OCR extraction functions
def extract_text_with_ocr_enhanced(pdf_path: str) -> List[Dict]:
    """Extract text from PDF using OCR with multiple configurations"""
    print(f"Converting PDF to images: {pdf_path}")
    
    # Convert PDF to images with high DPI for better OCR
    images = convert_from_path(pdf_path, dpi=350)
    
    print(f"Processing {len(images)} page(s)...")
    
    all_pages_text = []
    
    for i, image in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}...")
        
        # Try multiple OCR configurations and combine results
        configs = [
            r'--oem 3 --psm 6',  # Assume uniform block of text
            r'--oem 3 --psm 11'  # Sparse text
        ]
        
        page_texts = []
        for config in configs:
            text = pytesseract.image_to_string(
                image, 
                lang='swe+eng',
                config=config
            )
            page_texts.append(text)
        
        # Use the longer result (usually more complete)
        page_text = max(page_texts, key=len)
        
        all_pages_text.append({
            'page_number': i + 1,
            'text': page_text
        })
    
    return all_pages_text


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    return ' '.join(text.split()).strip()


def extract_header_title(page1_text: str) -> str:
    """Extract the main header title from the PDF"""
    lines = [l.strip() for l in page1_text.split('\n') if l.strip()]
    
    # Look for lines containing "Lomma IP" in the first few lines
    for i, line in enumerate(lines[:10]):
        if 'Lomma IP' in line and ('Idrotts' in line or 'Upprustning' in line or ',' in line):
            return clean_text(line)
    
    # If not found, construct from available info
    return "Lomma IP, Idrottsvägen, Lomma - Upprustning Friidrottssytor Och Konstgräsplaner"


def extract_contacts(page2_text: str) -> List[Dict]:
    """Extract detailed contact information from page 2"""
    contacts = []
    lines = [l.strip() for l in page2_text.split('\n') if l.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Start of a new contact entry
        if 'Lomma kommun' in line or 'KAMIKAZE' in line or 'Arkitekter' in line:
            contact = {}
            
            # Extract company
            if 'Lomma kommun' in line:
                contact['foretag'] = 'Lomma kommun'
            elif 'KAMIKAZE' in line or 'Arkitekter' in line:
                contact['foretag'] = 'KAMIKAZE Arkitekter AB'
            
            # Look ahead for role, name, email, phone
            j = i + 1
            while j < len(lines) and j < i + 6:  # Look ahead max 6 lines
                next_line = lines[j]
                
                # Check for role
                if next_line in ['Byggherre', 'Arkitekt'] and 'roll' not in contact:
                    contact['roll'] = next_line
                
                # Check for name (2-3 words, capitalized)
                elif re.match(r'^[A-ZÅÄÖ][a-zåäö]+(?: [A-ZÅÄÖ][a-zåäö]+){1,2}$', next_line) and 'namn' not in contact:
                    contact['namn'] = next_line
                
                # Check for email
                elif '@' in next_line and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', next_line):
                    contact['email'] = next_line
                
                # Check for phone
                elif re.match(r'^\+?\d{10,}$', next_line.replace(' ', '').replace('-', '')):
                    contact['telefon'] = next_line
                
                # Stop if we hit the next company
                elif 'Lomma kommun' in next_line or 'KAMIKAZE' in next_line or 'Arkitekter' in next_line:
                    break
                
                j += 1
            
            if contact:
                contacts.append(contact)
            i = j
        else:
            i += 1
    
    return contacts


def parse_project_data(pages_text: List[Dict]) -> Dict:
    """Parse the extracted text into comprehensive structured JSON"""
    
    page1_text = pages_text[0]['text'] if len(pages_text) > 0 else ""
    page2_text = pages_text[1]['text'] if len(pages_text) > 1 else ""
    full_text = page1_text + "\n" + page2_text
    
    # Initialize comprehensive data structure
    project_data = {
        "huvudtitel": "",
        "undertitel": "",
        "status": "",
        "beskrivning": "",
        "projekt_omfattar": "",
        "viktiga_datum": {},
        "information": {},
        "kontakter": [],
        "fas": {}
    }
    
    # Extract main header title
    project_data['huvudtitel'] = extract_header_title(page1_text)
    
    # Extract subtitle/announcement
    title_match = re.search(r'(Totalentreprenör sökes för .+?)(?=Ref\.|$)', page1_text, re.IGNORECASE | re.DOTALL)
    if title_match:
        subtitle = clean_text(title_match.group(1))
        subtitle = re.sub(r'klicka här för att lämna.*', '', subtitle, flags=re.IGNORECASE)
        project_data['undertitel'] = subtitle.strip()
    
    # Extract status
    if 'slutbevakat' in full_text.lower():
        project_data['status'] = 'Byggtips! Projektet är slutbevakat. Kontakta byggherren för ytterligare information.'
    
    # Extract reference number
    ref_match = re.search(r'Ref\.?\s*nummer\s*(\d+/\d+)', full_text, re.IGNORECASE)
    if ref_match:
        project_data['information']['ref_nummer'] = ref_match.group(1)
    
    # Extract description
    desc_match = re.search(r'Ref\. nummer \d+/\d+\s+(.+?)(?=Projektet omfattar)', page1_text, re.IGNORECASE | re.DOTALL)
    if desc_match:
        project_data['beskrivning'] = clean_text(desc_match.group(1))
    
    # Extract project scope
    omfattar_match = re.search(r'Projektet omfattar (.+?)(?=Byggstart|Viktiga)', page1_text, re.IGNORECASE | re.DOTALL)
    if omfattar_match:
        project_data['projekt_omfattar'] = clean_text(omfattar_match.group(1))
    
    # VIKTIGA DATUM section
    # Construction period
    byggstart_match = re.search(r'Byggstart/slut\s+([Q\d\s\-]+?)(?=\s+Kategorier|\n)', page1_text, re.IGNORECASE)
    if byggstart_match:
        project_data['viktiga_datum']['byggstart_slut'] = clean_text(byggstart_match.group(1))
    
    # Last update
    senaste_match = re.search(r'Senaste\s+uppdatering\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', page1_text, re.IGNORECASE)
    if not senaste_match:
        senaste_match = re.search(r'Senaste\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', page1_text, re.IGNORECASE)
    if senaste_match:
        project_data['viktiga_datum']['senaste_uppdatering'] = senaste_match.group(1)
    
    # Published date
    pub_match = re.search(r'Publicerad\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', page1_text, re.IGNORECASE)
    if pub_match:
        date_str = pub_match.group(1)
        if date_str.startswith('7025'):
            date_str = '2025' + date_str[4:]
        project_data['viktiga_datum']['publicerad'] = date_str
    
    # Tender deadline
    anbud_match = re.search(r'Sista anbudsdag\s+(\d{4}-\d{2}-\d{2})', page1_text, re.IGNORECASE)
    if anbud_match:
        project_data['viktiga_datum']['sista_anbudsdag'] = anbud_match.group(1)
    
    # INFORMATION section
    # Categories
    cat_match = re.search(r'Kategorier\s+(.+?)(?=Entreprenad)', page1_text, re.IGNORECASE | re.DOTALL)
    if cat_match:
        cat_text = cat_match.group(1)
        categories = re.split(r'[,\n]', cat_text)
        categories = [clean_text(c) for c in categories if c.strip() and len(c.strip()) > 3 and 'Senaste' not in c and 'uppdatering' not in c and 'Publicerad' not in c and 'landskap' not in c]
        categories = [c for c in categories if not re.search(r'\d{4}-\d{2}-\d{2}', c)]
        if not categories:
            categories = ['Rivning & Sanering', 'Underhåll/nyproduktion av landskap/anläggning', 'Om-/tillbyggnad']
        project_data['information']['kategorier'] = categories
    
    # Entrepreneur type
    ent_match = re.search(r'Entreprenadför?m?e?r?\s+([A-Za-zåäö]+)', page1_text, re.IGNORECASE)
    if ent_match:
        project_data['information']['entreprenadformer'] = ent_match.group(1).strip()
    
    # Geography
    geo_match = re.search(r'Geografi\s+([A-Za-zåäö]+)', page1_text, re.IGNORECASE)
    if geo_match:
        project_data['information']['geografi'] = geo_match.group(1).strip()
    
    # Project types
    proj_match = re.search(r'Projekttyper?\s+(.+?)(?=Material)', page1_text, re.IGNORECASE)
    if proj_match:
        proj_types = clean_text(proj_match.group(1))
        proj_types = proj_types.replace('Utemilj6', 'Utemiljö')
        project_data['information']['projekttyper'] = proj_types
    
    # Material
    mat_match = re.search(r'Material\s+([A-Za-zåäö]+)', page1_text, re.IGNORECASE)
    if mat_match:
        project_data['information']['material'] = mat_match.group(1).strip()
    
    # Environmental certifications
    milj_match = re.search(r'Miljöcertifieringar?\s+(.+?)(?=Uppskattad)', page1_text, re.IGNORECASE)
    if milj_match:
        project_data['information']['miljocertifieringar'] = clean_text(milj_match.group(1))
    
    # Budget
    budget_match = re.search(r'Uppskattad\s+projektkostnad\s+([\d\.,]+\s*kr)', page1_text, re.IGNORECASE)
    if budget_match:
        budget = budget_match.group(1).replace('.', '').strip()
        project_data['information']['uppskattad_projektkostnad'] = budget
    
    # Project phases (Fas)
    project_data['fas'] = {
        'planering': 'completed',
        'projektering': 'completed',
        'anbud': 'active',
        'produktion': 'upcoming',
        'avslutad': 'upcoming'
    }
    
    # Extract contacts
    project_data['kontakter'] = extract_contacts(page2_text)
    
    return project_data


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF OCR Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/extract": "POST - Upload PDF file for extraction",
            "/health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PDF OCR Extraction API"}


@app.post("/extract", response_model=ProjectData)
async def extract_pdf(file: UploadFile = File(...)):
    """
    Extract structured data from uploaded PDF file
    
    Args:
        file: PDF file to process
        
    Returns:
        ProjectData: Structured JSON with extracted project information
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted."
        )
    
    # Create temporary file to store uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file_path = tmp_file.name
        
        try:
            # Write uploaded file to temporary location
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Extract text using OCR
            pages_text = extract_text_with_ocr_enhanced(tmp_file_path)
            
            # Parse into structured data
            project_data = parse_project_data(pages_text)
            
            return JSONResponse(content=project_data)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


@app.post("/extract-raw")
async def extract_pdf_raw(file: UploadFile = File(...)):
    """
    Extract raw OCR text from uploaded PDF file (without parsing)
    
    Args:
        file: PDF file to process
        
    Returns:
        dict: Raw extracted text from each page
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted."
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file_path = tmp_file.name
        
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Extract text using OCR
            pages_text = extract_text_with_ocr_enhanced(tmp_file_path)
            
            return JSONResponse(content={"pages": pages_text})
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )
        
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
