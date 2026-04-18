import asyncio
import json
import re
import os
import io
import tempfile

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from playwright.async_api import async_playwright
from openai import OpenAI

app = FastAPI(title="Fee Scraper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

STANDARDIZE_PROMPT = """
You are a course name standardization engine.

Your task is to convert raw course names into a strict standardized format.

RULES:

1. Format:
   Full Degree Name [Short Form] (Specialization)

2. Honors:
   Use {Hons.} if mentioned

3. Integrated Courses:
   Use:
   Degree1 [Short1] + Degree2 [Short2]

4. Law Courses:
   Format:
   Bachelor of X + Bachelor of Laws [SHORTFORM]

5. Specialization:
   "/" or "," → split into multiple courses
   "&" → keep combined

6. Expand:
   BBA → Bachelor of Business Administration [BBA]
   BSc → Bachelor of Science [B.Sc]
   MSc → Master of Science [M.Sc]
   BTech → Bachelor of Technology [B.Tech]
   MTech → Master of Technology [M.Tech]
   BCA → Bachelor of Computer Applications [BCA]
   MBA → Master of Business Administration [MBA]
   BA → Bachelor of Arts [BA]
   BDesign → Bachelor of Design [B.Des]
   LLM → Master of Laws [LL.M]
   PhD → Doctor of Philosophy [Ph.D]

7. AI ML → Artificial Intelligence & Machine Learning

8. Remove symbols like *, extra spaces

OUTPUT:
Return ONLY JSON array.

Example:
["Bachelor of Science [B.Sc] (Chemistry)"]
"""


# ── Stage 1: Scrape ──────────────────────────────────────────────────────────

async def scrape_structured_data(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            tables_data = await page.evaluate("""() => {
                const results = [];
                const tables = document.querySelectorAll('table');
                tables.forEach(table => {
                    const rows = Array.from(table.querySelectorAll('tr'));
                    const matrix = rows.map(row =>
                        Array.from(row.querySelectorAll('th, td')).map(cell => cell.innerText.trim())
                    );
                    results.push(matrix);
                });
                return results;
            }""")
            return tables_data
        finally:
            await browser.close()


def tables_to_df(tables_data):
    valid_keywords = ["#", "sr", "s.no", "course", "program"]
    frames = []
    for table in tables_data:
        if len(table) > 1:
            headers = [col.replace('\n', ' ').strip() for col in table[0]]
            df = pd.DataFrame(table[1:], columns=headers)
            header_str = " ".join(headers).lower()
            if any(kw in header_str for kw in valid_keywords):
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True).dropna(how='all')


# ── Stage 2: Standardize ─────────────────────────────────────────────────────

def standardize_course(course_name: str, client: OpenAI) -> list:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": STANDARDIZE_PROMPT},
                {"role": "user", "content": course_name}
            ],
        )
        output = response.choices[0].message.content.strip()
        return json.loads(output)
    except Exception as e:
        print(f"Standardize error for '{course_name}': {e}")
        return [course_name]


def standardize_df(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    program_col = "Program" if "Program" in df.columns else (
        "Course Name:" if "Course Name:" in df.columns else df.columns[1]
    )
    all_rows = []
    for _, row in df.iterrows():
        courses = standardize_course(str(row[program_col]), client)
        for course in courses:
            new_row = row.copy()
            new_row[program_col] = course
            all_rows.append(new_row)
    return pd.DataFrame(all_rows)


# ── Stage 3: DB Pipeline ─────────────────────────────────────────────────────

def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text)).strip()


def match_course(raw_course, db_df):
    raw_norm = normalize(raw_course)
    for _, row in db_df.iterrows():
        if raw_norm == normalize(row["course_name"]):
            return row
    for _, row in db_df.iterrows():
        db_norm = normalize(row["course_name"])
        if raw_norm in db_norm or db_norm in raw_norm:
            return row
    return None


def get_val(row, col_name):
    val = row.get(col_name, 0)
    if pd.isna(val) or str(val).strip() == "":
        return 0
    clean_val = re.sub(r'[^\d]', '', str(val))
    return int(clean_val) if clean_val else 0


def run_db_pipeline(raw_df: pd.DataFrame, db_df: pd.DataFrame) -> pd.DataFrame:
    db_df = db_df.drop_duplicates(subset=["course_id"])

    # Dynamic college_id
    if "college_id" in db_df.columns:
        college_id = db_df["college_id"].iloc[0]
    else:
        college_id = 0

    final_rows = []
    for _, row in raw_df.iterrows():
        raw_program = row.get("Course Name:", row.get("Program", "Unknown"))
        db_match = match_course(raw_program, db_df)

        if db_match is not None:
            course_id = db_match["course_id"]
            tag = db_match["course_tag"]
            tag_name = db_match["course_tag_name"]
            final_duration = int(db_match["duration_years"])
        else:
            course_id, tag, tag_name = None, None, None
            final_duration = 1

        base_tuition = get_val(row, "Tution Fee: (Per Year)")
        admin_fee = get_val(row, "Administrative fee (per year)")
        security_fee = get_val(row, "Security (One time)")

        for year_idx in range(1, final_duration + 1):
            other_fees_list = []
            if admin_fee > 0:
                other_fees_list.append({"administrative_fee": str(admin_fee)})
            if year_idx == 1 and security_fee > 0:
                other_fees_list.append({"security_fee": str(security_fee)})

            row_total = base_tuition + admin_fee + (security_fee if year_idx == 1 else 0)

            final_rows.append({
                "college_id": college_id,
                "course_tag": tag,
                "course_tag_name": tag_name,
                "course_id": course_id,
                "course_name": raw_program,
                "year_added": 2026,
                "type": "year",
                "duration": year_idx,
                "quota": "general",
                "tuition_fee": base_tuition,
                "other_fees": json.dumps(other_fees_list),
                "total_fee": row_total,
                "application_fee": 0,
                "options_for_fee_source": "website",
                "is_fees_tentative": 0,
                "is_discontinued": 0
            })

    return pd.DataFrame(final_rows)


# ── Main Endpoint ─────────────────────────────────────────────────────────────

@app.post("/process")
async def process(
    fee_url: str = Form(...),
    db_master: UploadFile = File(...),
):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")

    # Load db_master
    db_bytes = await db_master.read()
    try:
        db_df = pd.read_excel(io.BytesIO(db_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read db_master.xlsx: {e}")

    # Stage 1: Scrape
    try:
        tables_data = await scrape_structured_data(fee_url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Scraping failed: {e}")

    scraped_df = tables_to_df(tables_data)
    if scraped_df.empty:
        raise HTTPException(status_code=422, detail="No fee tables found at that URL.")

    # Stage 2: Standardize
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
    std_df = standardize_df(scraped_df, oai_client)

    # Stage 3: DB pipeline
    final_df = run_db_pipeline(std_df, db_df)

    # Derive output filename from college_id
    college_id = final_df["college_id"].iloc[0] if not final_df.empty else "output"
    filename = f"{college_id}_fees.xlsx"

    # Stream back as Excel
    output = io.BytesIO()
    final_df.to_excel(output, index=False, sheet_name="FeeData")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/health")
def health():
    return {"status": "ok"}
