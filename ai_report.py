
import os
import time
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAMES = [
    "gemini-3.5-flash-lite"
]


REQUIRED_HEADINGS = [
    "1. Essential Facilities",
    "2. Connectivity",
    "3. Healthcare and Education",
    "4. Shopping and Lifestyle",
    "5. Strengths",
    "6. Possible Limitations",
    "7. Final Rental Assessment",
]


@st.cache_data(ttl=600, show_spinner=False)
def generate_neighbourhood_report(location, travel_mode, facility_data):

    if not API_KEY:
        raise ValueError(
            "Gemini API key not found. Check your .env file."
        )

    facility_data = "\n".join(
        str(facility_data).splitlines()[:30]
    )[:5000]

    prompt = f"""
You are an expert neighbourhood analysis assistant.

Analyse the supplied nearby facility information.

Location: {location}
Distance mode: {travel_mode}

Nearby facilities:
{facility_data}

Write a complete neighbourhood analysis report for someone
who is deciding whether to rent a house in this location.

IMPORTANT: Include ALL SEVEN sections below.
Do not stop after the first section.

1. Essential Facilities
Discuss nearby everyday facilities.

2. Connectivity
Discuss available transport, roads and connectivity
based only on the supplied data.

3. Healthcare and Education
Discuss nearby hospitals, clinics and educational facilities.

4. Shopping and Lifestyle
Discuss available shops, food facilities, parks and recreation.

5. Strengths
Explain the advantages supported by the supplied data.

6. Possible Limitations
Mention facilities that are missing or not represented
in the supplied data.

7. Final Rental Assessment
Give a balanced conclusion based only on the available data.

Rules:
- Include all seven headings.
- Write 2 to 4 sentences under each heading.
- Use only the supplied facility information.
- Never invent facility names or distances.
- Clearly mention when information is unavailable.
- Do not make unsupported safety claims.
- Keep the complete report below 600 words.
"""

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=120000)
    )

    errors = []

    try:

        for model in MODEL_NAMES:

            for attempt in range(2):

                try:

                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            max_output_tokens=2000
                        )
                    )

                    report = response.text or ""

                    missing_headings = [
                        heading
                        for heading in REQUIRED_HEADINGS
                        if heading.lower() not in report.lower()
                    ]

                    if not report.strip() or missing_headings:
                        raise RuntimeError(
                            "Incomplete report. Missing sections: "
                            + ", ".join(missing_headings)
                        )

                    return report

                except Exception as error:

                    error_message = str(error)

                    errors.append(
                        f"{model}: {error_message}"
                    )

                    temporary_error = any(
                        code in error_message.upper()
                        for code in [
                            "503",
                            "UNAVAILABLE",
                            "429",
                            "RESOURCE_EXHAUSTED",
                            "500",
                            "504",
                            "INTERNAL",
                            "INCOMPLETE REPORT",
                        ]
                    )

                    if not temporary_error:
                        raise RuntimeError(
                            f"Gemini report error: {error_message}"
                        ) from error

                    if attempt == 0:
                        time.sleep(3)

        raise RuntimeError(
            "Gemini could not generate a complete report "
            "after trying the available models.\n\n"
            + "\n".join(errors)
        )

    finally:
        client.close()
