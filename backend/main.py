import io
import os
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image
import sympy as sp

app = FastAPI(title="ShikshaAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def solve_math_expression(expression: str) -> str:
    try:
        expr = sp.sympify(expression)
        return f"Verified Result: {sp.simplify(expr)}"
    except Exception as e:
        return f"Math check note: {e}"

GRADE_STYLES = {
    "1-5": "ધોરણ ૧ થી ૫ ના બાળકને એકદમ સરળ શબ્દો, નાના વાક્યો અને રોજિંદા ઉદાહરણો (સફરજન, રમકડાં, સિક્કા) આપીને નાના સ્ટેપ્સમાં સમજાવો.",
    "6-10": "ધોરણ ૬ થી ૧૦ ના વિદ્યાર્થીને પાઠ્યપુસ્તક (GSEB/CBSE) ની રીત મુજબ સૂત્રો અને દરેક સ્ટેપ સ્પષ્ટ રીતે સમજાવો.",
    "11-12": "ધોરણ ૧૧-૧૨ (Science/Entrance) ના વિદ્યાર્થીને મૂળ સિદ્ધાંતો, SI એકમો અને ચોક્કસ ગણતરી સાથે સમજાવો.",
    "college": "કોલેજના વિદ્યાર્થીને ચોક્કસ વ્યાખ્યાઓ, સાબિતી, ટેકનિકલ ઊંડાણ અને એજ-કેસ સાથે સમજાવો."
}

@app.post("/api/ask")
async def ask_tutor(
    grade: str = Form(...),
    subject: str = Form(...),
    language: str = Form("Gujarati"),
    mode: str = Form("hint"),
    question: str = Form(""),
    image: Optional[UploadFile] = File(None)
):
    style = GRADE_STYLES.get(grade, GRADE_STYLES["6-10"])
    mode_text = (
        "પહેલા એક મદદરૂપ Hint અને પહેલું સ્ટેપ જ આપો, પછી વિદ્યાર્થીને આગળનો પ્રયત્ન કરવા કહો."
        if mode == "hint"
        else "આખો જવાબ સ્ટેપ-બાય-સ્ટેપ સ્પષ્ટ રીતે સમજાવો."
    )

    system_instruction = (
        f"{style}\n"
        f"વિષય: {subject}\n"
        f"જવાબની ભાષા: {language}\n"
        f"ભણાવવાની રીત: {mode_text}\n"
        "જ્યારે પણ ગણતરી કે બીજગણિત આવે ત્યારે solve_math_expression ટૂલનો ઉપયોગ કરીને ચકાસો."
    )

    contents = []
    if image:
        img_bytes = await image.read()
        contents.append(Image.open(io.BytesIO(img_bytes)))

    prompt_text = question.strip() if question.strip() else "ફોટામાં આપેલો સવાલ ઉકેલીને સમજાવો."
    contents.append(prompt_text)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[solve_math_expression],
            temperature=0.3,
        ),
    )

    return {"answer": response.text}
  
