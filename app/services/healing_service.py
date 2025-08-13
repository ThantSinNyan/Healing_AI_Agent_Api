import json
import os
import re
import google.generativeai as genai

def generate_healing(birth_date: str, time: str, birth_place: str, language: str) -> dict:
    # Load API key from environment variable (should be secured this way in production)
    api_key = os.getenv("GOOGLE_GENAI_API_KEY") or "AIzaSyAYFdHysdBYMoMb64YyJCPw2nBWjKLdOsQ"
    if not api_key:
        raise ValueError("Google Generative AI API key not found in environment variables.")

    genai.configure(api_key=api_key)

    prompt_template = """
You are a trauma-informed, astrology-aware healing guide.

Input Data:
- Birth Date: {birth_date}
- Birth Time: {time}
- Birth Place: {birth_place}

You have to focus on the User's Chiron Placement: Chiron in zodiac_sign in the house.

🗣️ Language Preference: Please generate the entire response in **{language}**.

Translate gently and contextually — not word-for-word — while keeping the meaning, warmth, and emotional depth.
Ensure the tone remains poetic, empathetic, trauma-sensitive, and culturally natural in {language}.

Generate a complete healing journey based on this placement.
Make sure your response has the following structure:
{{
    "mainTitle": "Chiron in Scorpio in the 4th House",
    "description": "A reflective overview of this placement’s emotional and spiritual themes.",

    "CoreWoundsAndEmotionalThemes": ["keywords that capture deep emotional wounds, e.g., abandonment, betrayal, etc."],
    "PatternsAndStruggles": ["keywords that reflect common behavioral or emotional struggles."],
    "HealingAndTransformation": ["keywords that represent the healing path and emotional growth."],
    "SpiritualWisdomAndGifts": ["keywords showing the spiritual gifts and insights gained through healing."],

    "woundPoints": ["Write 3–4 emotional facts or experiences that often come with this Chiron placement."],

    "PatternsConnectedToThisWound": ["Describe 3–4 behavioral or relational patterns that are shaped by this wound."],

    "Healing Benefits": ["List 3–4 healing outcomes — personal growth, peace, transformation — that come from facing and healing this wound."]
}}

💡 Tone & Style:
- Safe and reflective
- Poetic, warm, empowering
- Gentle and trauma-aware
- Emotionally intelligent and culturally appropriate in {language}

Notes for writing:
- For `description`: Provide a brief but emotionally deep overview of the Chiron wound and healing potential.
- For keyword sections: Only list **relevant themes as short phrases**.
- For bullet-point sections: Create **natural-sounding, insightful sentences** that reflect lived emotional experience.
"""
    prompt = prompt_template.format(
        birth_date=birth_date,
        time=time,
        birth_place=birth_place,
        language=language
    )

    # Load and call the Gemini model
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    result_text = response.text.strip()
    print(result_text)
    json_match = re.search(r'\{[\s\S]*?\}', result_text)

    if json_match:
        json_str = json_match.group(0)
    else:
        raise ValueError("Could not find a valid JSON object in model response")
    healing_data = json.loads(json_str)

    return healing_data
