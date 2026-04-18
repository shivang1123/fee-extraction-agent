import json
import re
import os
import io
import time
from difflib import SequenceMatcher

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from playwright.async_api import async_playwright
from openai import OpenAI, RateLimitError, APIError

app = FastAPI(title="Fee Scraper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o-mini"

FEE_COLUMN_ALIASES = {
    "tuition":  ["tution fee: (per year)", "tuition fee (per year)", "tuition fee",
                 "per year fee", "course fee", "per semester", "fees per semester",
                 "annual fee", "per annum", "yearly fee", "fee per year",
                 "total course fee", "total fee"],
    "admin":    ["administrative fee (per year)", "admin fee", "administrative fee",
                 "management fee", "development fee"],
    "security": ["security (one time)", "security deposit", "security fee",
                 "caution money", "refundable deposit"],
    "program":  ["program", "course name:", "course name", "programme",
                 "courses offered", "course", "name of course", "subject"],
}

STANDARDIZE_PROMPT = """
You are a course name standardization engine.
Convert raw course names into strict standardized format.

RULES:
1. Format: Full Degree Name [Short Form] (Specialization)
2. Honors: Use {Hons.} if mentioned
3. Integrated: Degree1 [Short1] + Degree2 [Short2]
4. Specialization: "/" or "," split into MULTIPLE array entries; "&" keep combined
5. Expand abbreviations:
   BBA->Bachelor of Business Administration [BBA]
   BSc/B.Sc->Bachelor of Science [B.Sc]
   MSc/M.Sc->Master of Science [M.Sc]
   BTech/B.Tech->Bachelor of Technology [B.Tech]
   MTech/M.Tech->Master of Technology [M.Tech]
   BCA->Bachelor of Computer Applications [BCA]
   MCA->Master of Computer Applications [MCA]
   MBA->Master of Business Administration [MBA]
   BA->Bachelor of Arts [BA]
   MA->Master of Arts [MA]
   BCom/B.Com->Bachelor of Commerce [B.Com]
   MCom/M.Com->Master of Commerce [M.Com]
   BDesign/B.Des->Bachelor of Design [B.Des]
   LLB->Bachelor of Laws [LL.B]
   LLM->Master of Laws [LL.M]
   PhD/Ph.D->Doctor of Philosophy [Ph.D]
   BPharma->Bachelor of Pharmacy [B.Pharm]
   MPharma->Master of Pharmacy [M.Pharm]
   MBBS->Bachelor of Medicine and Surgery [MBBS]
   BDS->Bachelor of Dental Surgery [BDS]
   BEd->Bachelor of Education [B.Ed]
   MEd->Master of Education [M.Ed]
6. AI ML -> Artificial Intelligence & Machine Learning
7. Remove *, extra spaces, trailing punctuation
8. If input is not a course name (e.g. "Total", "Note:"), return []

OUTPUT: ONLY a JSON array of strings. No markdown.
"""

AI_EXTRACTION_PROMPT = """
You are a fee data extraction engine. Extract ALL courses and fees from the given college page content.
Return a JSON array of objects, each with:
- "program": course name (string)
- "tuition_fee": annual tuition as integer (0 if not found)
- "admin_fee": admin/management fee per year as integer (0 if not found)
- "security_fee": one-time security deposit as integer (0 if not found)
- "duration_sem": duration in semesters as integer (0 if unknown)
- "fee_type": "annual" or "semester"

Rules: strip commas/Rs/currency symbols from numbers. Return [] if no data. ONLY JSON.
"""


async def scrape_structured_data(url: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            viewport={"width": 1280, "height": 900},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
        page = await context.new_page()
        try:
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
            except Exception:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(3000)

            tables = await page.evaluate("""() => {
                function extractTable(table) {
                    return Array.from(table.querySelectorAll('tr')).map(row =>
                        Array.from(row.querySelectorAll('th, td'))
                            .map(cell => cell.innerText.replace(/\\s+/g, ' ').trim())
                    ).filter(r => r.some(c => c.length > 0));
                }
                return Array.from(document.querySelectorAll('table')).map(extractTable);
            }""")

            iframe_tables = []
            for frame in page.frames:
                if frame == page.main_frame:
                    continue
                try:
                    ft = await frame.evaluate("""() => {
                        return Array.from(document.querySelectorAll('table')).map(t =>
                            Array.from(t.querySelectorAll('tr')).map(row =>
                                Array.from(row.querySelectorAll('th,td'))
                                    .map(c => c.innerText.replace(/\\s+/g,' ').trim())
                            ).filter(r => r.some(c => c.length > 0))
                        );
                    }""")
                    iframe_tables.extend(ft)
                except Exception:
                    pass

            page_text = await page.evaluate("""() => {
                const clone = document.cloneNode(true);
                clone.querySelectorAll('script,style,nav,footer,header').forEach(e=>e.remove());
                return clone.body ? clone.body.innerText.replace(/\\s+/g,' ').trim() : '';
            }""")

            return {
                "tables":    tables + iframe_tables,
                "page_text": page_text[:8000],
            }
        finally:
            await browser.close()


def find_program_col(df: pd.DataFrame) -> str:
    aliases = FEE_COLUMN_ALIASES["program"]
    for col in df.columns:
        if col.lower().strip() in aliases:
            return col
    for col in df.columns:
        for alias in aliases:
            if alias in col.lower() or col.lower() in alias:
                return col
    return df.columns[0]


def find_fee_col(df: pd.DataFrame, fee_type: str):
    aliases = FEE_COLUMN_ALIASES.get(fee_type, [])
    for col in df.columns:
        col_l = col.lower().strip()
        if col_l in aliases:
            return col
        for alias in aliases:
            if alias in col_l or col_l in alias:
                return col
    return None


def ai_extract_fees(content: str, client: OpenAI) -> list:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":AI_EXTRACTION_PROMPT},
                      {"role":"user","content":content}],
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"AI extraction failed: {e}")
        return []


def tables_to_df(scrape_result: dict, oai_client: OpenAI) -> pd.DataFrame:
    tables    = scrape_result.get("tables", [])
    page_text = scrape_result.get("page_text", "")
    fee_kw    = ["fee","tuition","course","program","programme","admission","annual","#","sr","s.no"]
    bad_kw    = ["menu","nav","login","contact","footer","social"]

    frames = []
    for table in tables:
        if len(table) < 2:
            continue
        headers = [str(c).replace('\n',' ').strip() for c in table[0]]
        hstr = " ".join(headers).lower()
        if not any(k in hstr for k in fee_kw):
            continue
        if any(k in hstr for k in bad_kw):
            continue
        try:
            df = pd.DataFrame(table[1:], columns=headers)
            df = df.dropna(how='all')
            frames.append(df)
        except Exception:
            pass

    if frames:
        return pd.concat(frames, axis=0, ignore_index=True)

    print("No tables — using AI extraction fallback")
    ai_rows = ai_extract_fees(page_text or "", oai_client)
    return pd.DataFrame(ai_rows) if ai_rows else pd.DataFrame()


def standardize_course_with_retry(name: str, client: OpenAI, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":STANDARDIZE_PROMPT},
                          {"role":"user","content":name}],
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
            result = json.loads(raw)
            return result if isinstance(result, list) else [name]
        except RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Standardize attempt {attempt+1} error: {e}")
            time.sleep(1)
    return [name]


def standardize_df(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    pcol  = find_program_col(df)
    cache = {}
    rows  = []
    for _, row in df.iterrows():
        name = str(row.get(pcol, "")).strip()
        if not name or name.lower() in ("nan","none",""):
            continue
        if name not in cache:
            cache[name] = standardize_course_with_retry(name, client)
        for std in cache[name]:
            nr = row.copy()
            nr[pcol] = std
            nr["_original_program"] = name
            rows.append(nr)
    return pd.DataFrame(rows) if rows else df


def normalize(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def match_course(raw: str, db_df: pd.DataFrame) -> tuple:
    raw_n = normalize(raw)
    if not raw_n:
        return None, 0.0, "none"
    best_row, best_score, best_type = None, 0.0, "none"
    stop = {"of","in","and","the","for","to","a","an"}

    for _, row in db_df.iterrows():
        db_n = normalize(row["course_name"])
        if not db_n:
            continue
        if raw_n == db_n:
            return row, 1.0, "exact"
        if raw_n in db_n or db_n in raw_n:
            score = len(min(raw_n,db_n,key=len)) / len(max(raw_n,db_n,key=len))
            if score > best_score:
                best_row, best_score, best_type = row, score, "substring"
            continue
        rt = set(raw_n.split()) - stop
        dt = set(db_n.split())  - stop
        if rt and dt:
            overlap = len(rt & dt) / len(rt | dt)
            if overlap > 0.6 and overlap > best_score:
                best_row, best_score, best_type = row, overlap, "token_overlap"
        fuzz = SequenceMatcher(None, raw_n, db_n).ratio()
        if fuzz > 0.82 and fuzz > best_score:
            best_row, best_score, best_type = row, fuzz, "fuzzy"

    return best_row, best_score, best_type


def get_val(row, col) -> int:
    if col is None:
        return 0
    val = row.get(col, 0)
    if pd.isna(val) or str(val).strip() in ("","nan"):
        return 0
    clean = re.sub(r'[^\d]', '', str(val))
    return int(clean) if clean else 0


def run_db_pipeline(raw_df: pd.DataFrame, db_df: pd.DataFrame) -> tuple:
    db_df = db_df.drop_duplicates(subset=["course_id"])
    college_id = db_df["college_id"].iloc[0] if "college_id" in db_df.columns else 0

    pcol         = find_program_col(raw_df)
    tuition_col  = find_fee_col(raw_df, "tuition")
    admin_col    = find_fee_col(raw_df, "admin")
    security_col = find_fee_col(raw_df, "security")
    has_ai       = "tuition_fee" in raw_df.columns

    cols_lower = " ".join(raw_df.columns).lower()
    fee_type   = "semester" if ("semester" in cols_lower or "per sem" in cols_lower) else "annual"
    if "_fee_type" in raw_df.columns and "semester" in raw_df["_fee_type"].values:
        fee_type = "semester"

    matched, unmatched = [], []

    for _, row in raw_df.iterrows():
        raw_prog = str(row.get("_original_program", row.get(pcol, ""))).strip()
        std_prog = str(row.get(pcol, raw_prog)).strip()
        if not raw_prog or raw_prog.lower() in ("nan","unknown",""):
            continue

        db_match, conf, mtype = match_course(std_prog, db_df)

        if has_ai:
            tuition  = int(row.get("tuition_fee", 0) or 0)
            admin    = int(row.get("admin_fee", 0) or 0)
            security = int(row.get("security_fee", 0) or 0)
            ai_ftype = str(row.get("fee_type", fee_type))
        else:
            tuition  = get_val(row, tuition_col)
            admin    = get_val(row, admin_col)
            security = get_val(row, security_col)
            ai_ftype = fee_type

        if ai_ftype == "semester" and tuition > 0:
            tuition = tuition * 2
            admin   = admin * 2

        if db_match is not None:
            dur = int(db_match["duration_years"])
            for yr in range(1, dur + 1):
                other = []
                if admin > 0:
                    other.append({"administrative_fee": str(admin)})
                if yr == 1 and security > 0:
                    other.append({"security_fee": str(security)})
                total = tuition + admin + (security if yr == 1 else 0)
                matched.append({
                    "college_id":             college_id,
                    "course_tag":             db_match["course_tag"],
                    "course_tag_name":        db_match["course_tag_name"],
                    "course_id":              db_match["course_id"],
                    "course_name":            std_prog,
                    "year_added":             2026,
                    "type":                   "year",
                    "duration":               yr,
                    "quota":                  "general",
                    "tuition_fee":            tuition,
                    "other_fees":             json.dumps(other),
                    "total_fee":              total,
                    "application_fee":        0,
                    "options_for_fee_source": "website",
                    "is_fees_tentative":      0,
                    "is_discontinued":        0,
                    "_match_type":            mtype,
                    "_match_confidence":      round(conf, 2),
                })
        else:
            unmatched.append({
                "raw_course_name":    raw_prog,
                "standardized_name":  std_prog,
                "tuition_fee_found":  tuition,
                "admin_fee_found":    admin,
                "security_fee_found": security,
                "reason":             "No match found in DB master",
                "action_needed":      "Map manually to course_id",
            })

    return pd.DataFrame(matched), pd.DataFrame(unmatched)


@app.post("/process")
async def process(
    fee_url:   str        = Form(...),
    db_master: UploadFile = File(...),
):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not configured on server.")

    db_bytes = await db_master.read()
    try:
        db_df = pd.read_excel(io.BytesIO(db_bytes))
    except Exception as e:
        raise HTTPException(400, f"Cannot read master sheet: {e}")

    required = {"course_id","course_name","duration_years"}
    missing  = required - set(db_df.columns)
    if missing:
        raise HTTPException(400, f"Master sheet missing columns: {missing}")

    oai_client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        scrape_result = await scrape_structured_data(fee_url)
    except Exception as e:
        raise HTTPException(422, f"Scraping failed: {e}")

    scraped_df = tables_to_df(scrape_result, oai_client)
    if scraped_df.empty:
        raise HTTPException(422, "No fee data found. Try a more specific fee page URL.")

    std_df = standardize_df(scraped_df, oai_client)
    matched_df, unmatched_df = run_db_pipeline(std_df, db_df)

    if matched_df.empty and unmatched_df.empty:
        raise HTTPException(422, "Pipeline produced no output.")

    college_id = matched_df["college_id"].iloc[0] if not matched_df.empty else "output"
    filename   = f"{college_id}_fees.xlsx"

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if not matched_df.empty:
            export = matched_df.drop(columns=["_match_type","_match_confidence"], errors="ignore")
            export.to_excel(writer, index=False, sheet_name="FeeData")
            low = matched_df[matched_df["_match_confidence"] < 0.75]
            if not low.empty:
                low.to_excel(writer, index=False, sheet_name="ReviewMatches")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched")

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/health")
def health():
    return {"status": "ok"}
