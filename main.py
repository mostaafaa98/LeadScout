import asyncio
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urljoin

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from playwright.async_api import Browser, BrowserContext, Page, async_playwright


# ============================================================
# LeadScout / MABS
# Google Maps web extraction backend
# ============================================================

APP_NAME = "LeadScout"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
WEB_DIR = BASE_DIR / "web"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

# Default to installed Chrome because the debug run on this PC
# showed the Google Maps results loading correctly in Chrome.
BROWSER_CHANNEL = os.getenv("LEADSCOUT_BROWSER", "chrome").strip() or "chrome"

# Use visible browser by default during development so failures
# are easy to inspect. Set LEADSCOUT_HEADLESS=1 for headless mode.
HEADLESS = os.getenv("LEADSCOUT_HEADLESS", "0") == "1"

PORT = int(os.getenv("LEADSCOUT_PORT", "8000"))

MAX_LOCATION_CONCURRENCY = int(os.getenv("LEADSCOUT_LOCATION_CONCURRENCY", "2"))
DETAIL_CONCURRENCY = int(os.getenv("LEADSCOUT_DETAIL_CONCURRENCY", "4"))

app = FastAPI(
    title="LeadScout API",
    version="1.1.0",
    description="Google Maps web extraction backend without Google Places API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Models
# ============================================================

class ScrapeRequest(BaseModel):
    category: str = Field(..., min_length=1, max_length=200)
    location: Optional[str] = Field(default=None, max_length=200)  # backward compatibility
    locations: list[str] = Field(default_factory=list, max_length=20)
    max_results: int = Field(default=20, ge=1, le=500)


@dataclass
class ScrapeJob:
    job_id: str
    category: str
    locations: list[str]
    target_per_location: int

    state: str = "idle"
    message: str = "جاهز"
    error: Optional[str] = None

    found: int = 0
    processed: int = 0

    # Per-location progress
    location_status: dict[str, dict] = field(default_factory=dict)

    # URLs collected from the search list
    link_items: list[dict] = field(default_factory=list)

    # Final structured leads
    results: list[dict] = field(default_factory=list)

    seen_result_keys: set[str] = field(default_factory=set)
    seen_link_keys: set[str] = field(default_factory=set)

    output_file: Optional[str] = None
    task: Optional[asyncio.Task] = None

    # Event set = paused, clear = running
    pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    page: Optional[Page] = None


jobs: dict[str, ScrapeJob] = {}
active_job_id: Optional[str] = None


# ============================================================
# Text helpers
# ============================================================

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def normalize_digits(value: str) -> str:
    return (value or "").translate(ARABIC_DIGITS)


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""

    value = value.replace("\u200e", " ")
    value = value.replace("\u200f", " ")
    value = value.replace("\u202a", " ")
    value = value.replace("\u202b", " ")
    value = value.replace("\u202c", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_key(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^\w\u0600-\u06ff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def safe_filename(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = value.replace(" ", "_")
    return value[:80] or "search"


def _normalize_phone_candidate(candidate: str) -> str:
    """Normalize one phone-like candidate and reject rating-like text."""
    if not candidate:
        return ""

    candidate = normalize_digits(candidate)
    candidate = candidate.replace("\u200e", " ").replace("\u200f", " ")
    candidate = candidate.strip()

    # Common accessibility text for ratings such as: 5.0 1 2 3 4 5
    if re.match(r"^\s*[0-5][.,]\d(?:\s+[0-5]){2,}\s*$", candidate):
        return ""

    # A decimal rating by itself is never a phone number.
    if re.fullmatch(r"\s*[0-5][.,]\d\s*", candidate):
        return ""

    # Google Maps sometimes exposes RTL phone text with the + sign at the end:
    # 964787077700+  -> +964787077700
    candidate = candidate.replace("\u202a", "").replace("\u202b", "").replace("\u202c", "")
    if candidate.count("+") == 1 and candidate.rstrip().endswith("+"):
        candidate = "+" + candidate.rstrip()[:-1].strip()

    digits = re.sub(r"\D", "", candidate)

    if not 7 <= len(digits) <= 15:
        return ""

    # Reject strings that look like accessibility rating scales:
    # 1 2 3 4 5 / 5 1 2 3 4 5 / 4.5 1 2 3 4 5
    compact_groups = re.findall(r"\d", candidate)
    if len(compact_groups) >= 5:
        single_digit_groups = re.findall(r"(?<!\d)\d(?!\d)", candidate)
        if len(single_digit_groups) >= 4 and not re.search(r"[+-]?\d[\d\s().-]{7,}", candidate):
            return ""

    # Preserve international +. For bare international country codes,
    # make the output consistent when the code is recognizable.
    if candidate.startswith("+"):
        return "+" + digits

    known_country_codes = (
        "20", "218", "216", "212", "213", "218", "249", "961", "962", "963",
        "964", "965", "966", "967", "968", "971", "972", "973", "974", "975",
        "976", "90", "44", "33", "49", "39", "34", "31", "32", "1",
    )

    if digits.startswith(known_country_codes):
        return "+" + digits

    return candidate


def parse_phone(text: str) -> str:
    """Extract a plausible phone number without confusing ratings/review counts."""
    if not text:
        return ""

    text = normalize_digits(text)
    text = text.replace("\u200e", " ").replace("\u200f", " ")

    # First pass: line-by-line. Google Maps normally puts the phone on
    # its own line in the result card, which is much safer than scanning
    # the entire card as one string.
    for line in re.split(r"[\r\n]+", text):
        line = clean_text(line)
        if not line:
            continue

        # Avoid labels that are clearly review/rating metadata.
        if re.search(r"(?:مراجعات?|reviews?|نجوم?|stars?)", line, re.I):
            continue

        # Typical phone-like strings, allowing spaces, hyphens, brackets,
        # parentheses, and a misplaced trailing + from RTL rendering.
        candidates = re.findall(
            r"(?:\+?\d[\d\s().-]{6,}\d\+?)",
            line,
        )

        for candidate in candidates:
            phone = _normalize_phone_candidate(candidate)
            if phone:
                return phone

        # If the line itself is only a phone-like value, normalize it.
        if re.fullmatch(r"[+\d\s().-]{7,}", line):
            phone = _normalize_phone_candidate(line)
            if phone:
                return phone

    # Second pass: search for labeled phone values in Arabic/English.
    labeled = re.findall(
        r"(?:هاتف|اتصال|phone|tel|telephone)\s*[:：-]?\s*([+\d][\d\s().-]{6,}\d\+?)",
        text,
        flags=re.I,
    )

    for candidate in labeled:
        phone = _normalize_phone_candidate(candidate)
        if phone:
            return phone

    return ""


async def find_phone(
    page: Page,
    fallback_text: str = "",
) -> str:
    """
    Google Maps can expose the phone number as:
    - visible text
    - aria-label
    - title
    - data-item-id
    - tel: href
    Some icon buttons return a misleading inner_text like "0",
    so every candidate is validated before accepting it.
    """

    selectors = [
        'a[href^="tel:"]',
        '[data-item-id*="phone"]',
        'button[aria-label*="Phone"]',
        'button[aria-label*="phone"]',
        'button[aria-label*="هاتف"]',
        'button[aria-label*="اتصال"]',
        '[aria-label*="Phone"]',
        '[aria-label*="phone"]',
        '[aria-label*="هاتف"]',
        '[aria-label*="اتصال"]',
    ]

    attributes = [
        "href",
        "aria-label",
        "title",
        "data-item-id",
    ]

    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()

            for i in range(min(count, 6)):
                try:
                    el = loc.nth(i)

                    href_value = await el.get_attribute("href")
                    if href_value and href_value.lower().startswith("tel:"):
                        raw_tel = href_value[4:]
                        phone = _normalize_phone_candidate(raw_tel)
                        if phone:
                            return phone

                    for attr in attributes:
                        value = await el.get_attribute(attr)
                        phone = parse_phone(value or "")

                        if phone:
                            return phone

                    try:
                        value = await el.inner_text(timeout=1500)
                    except Exception:
                        value = ""

                    phone = parse_phone(value)

                    if phone:
                        return phone

                except Exception:
                    continue

        except Exception:
            continue

    try:
        body = clean_text(
            await page.locator("body").inner_text(timeout=8000)
        )

        phone = parse_phone(body)

        if phone:
            return phone

    except Exception:
        pass

    return parse_phone(fallback_text)


def parse_rating(text: str) -> Optional[float]:
    if not text:
        return None

    text = normalize_digits(text)

    patterns = [
        r"([0-5](?:[.,]\d)?)\s*\(\s*[\d,.]+\s*\)",
        r"([0-5](?:[.,]\d)?)\s*(?:نجوم?|نجمة|stars?)",
        r"★\s*([0-5](?:[.,]\d)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                continue

    return None


def parse_reviews(text: str) -> Optional[int]:
    if not text:
        return None

    text = normalize_digits(text)

    patterns = [
        r"\(([\d,.]+)\)",
        r"([\d,.]+)\s*(?:مراجعة|مراجعات|reviews?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            raw = match.group(1).replace(",", "").replace(".", "")

            try:
                return int(raw)
            except ValueError:
                continue

    return None


# ============================================================
# Generic Playwright helpers
# ============================================================

async def first_text(page: Page, selectors: list[str], timeout: int = 2500) -> str:
    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()

            if not count:
                continue

            for i in range(min(count, 3)):
                try:
                    value = clean_text(
                        await loc.nth(i).inner_text(timeout=timeout)
                    )
                    if value:
                        return value
                except Exception:
                    continue

        except Exception:
            continue

    return ""


async def first_attr(
    page: Page,
    selectors: list[str],
    attr: str,
    timeout: int = 2500,
) -> str:
    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()

            if not count:
                continue

            for i in range(min(count, 3)):
                try:
                    value = clean_text(
                        await loc.nth(i).get_attribute(attr, timeout=timeout)
                    )

                    if value:
                        return value
                except Exception:
                    continue

        except Exception:
            continue

    return ""


async def click_if_visible(page: Page, selectors: list[str]) -> bool:
    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()

            if not count:
                continue

            first = loc.first

            if await first.is_visible(timeout=1200):
                await first.click(timeout=3000)
                await page.wait_for_timeout(500)
                return True

        except Exception:
            continue

    return False


async def accept_google_dialogs(page: Page) -> None:
    await click_if_visible(
        page,
        [
            'button:has-text("قبول الكل")',
            'button:has-text("أوافق")',
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            '[aria-label="قبول الكل"]',
            '[aria-label="Accept all"]',
        ],
    )


async def wait_for_result_list(page: Page, timeout_ms: int = 90000) -> int:
    """
    Google Maps can take a long time to populate the search list
    on a slow connection. The debug run on this machine showed the
    list arriving only after a long wait, so this function polls for
    actual result cards instead of failing early.
    """
    deadline = asyncio.get_running_loop().time() + (timeout_ms / 1000)

    selectors = [
        'div[role="feed"] div[role="article"]',
        'div[role="article"]',
        'a[href*="/maps/place/"]',
    ]

    while asyncio.get_running_loop().time() < deadline:
        await accept_google_dialogs(page)

        for selector in selectors:
            try:
                count = await page.locator(selector).count()
                if count > 0:
                    return count
            except Exception:
                pass

        await page.wait_for_timeout(1000)

    return 0


# ============================================================
# Search workflow
# ============================================================

async def open_search_and_collect(
    job: ScrapeJob,
    page: Page,
    search_location: str,
) -> list[dict]:
    """
    Search through the Google Maps UI itself rather than relying
    solely on a direct /maps/search URL. This matches the working
    debug flow seen on the development PC.
    """

    job.state = "running"
    job.message = "جاري فتح Google Maps..."

    await page.goto(
        "https://www.google.com/maps?hl=ar&gl=eg",
        wait_until="domcontentloaded",
        timeout=60000,
    )

    # Initial page bootstrap can be slow.
    await page.wait_for_timeout(5000)
    await accept_google_dialogs(page)

    search_box = page.locator(
        'input[name="q"]'
    ).first

    # The current Google Maps DOM exposed input[name=q] in the debug run.
    await search_box.wait_for(
        state="visible",
        timeout=30000,
    )

    query = f"{job.category} في {search_location}"

    job.message = f"جاري البحث عن: {query}"

    await search_box.fill(query)

    search_button = page.locator(
        'button[aria-label="بحث"]'
    ).first

    if await search_button.count():
        try:
            await search_button.click(timeout=5000)
        except Exception:
            await search_box.press("Enter")
    else:
        await search_box.press("Enter")

    # Wait for Google's results network/DOM to settle.
    job.message = "جاري انتظار تحميل نتائج Google Maps..."

    count = await wait_for_result_list(
        page,
        timeout_ms=90000,
    )

    if count <= 0:
        # One final wait after the polling cycle.
        await page.wait_for_timeout(5000)
        count = await page.locator(
            'div[role="article"]'
        ).count()

    if count <= 0:
        raise RuntimeError(
            "Google Maps لم يعرض قائمة نتائج داخل الصفحة بعد الانتظار. "
            "اترك المتصفح مفتوحًا أثناء الاختبار واضبط LEADSCOUT_HEADLESS=0 "
            "لمراجعة ما يظهر بصريًا."
        )

    job.location_status.setdefault(
        search_location,
        {
            "state": "running",
            "found": 0,
            "processed": 0,
            "target": job.target_per_location,
            "message": "",
        },
    )

    results = await collect_search_cards(
        job,
        page,
        search_location,
    )

    job.location_status[search_location]["found"] = len(results)
    job.location_status[search_location]["message"] = (
        f"تم جمع {len(results)} نتيجة — جاري تجهيز التفاصيل"
    )

    return results


async def collect_search_cards(
    job: ScrapeJob,
    page: Page,
    search_location: str,
) -> list[dict]:
    """
    Collect result cards. The DOM can change between searches, so we
    inspect both article cards and /maps/place/ links.
    """

    collected: list[dict] = []

    stable_rounds = 0
    previous_count = 0

    max_rounds = max(
        20,
        min(
            150,
            job.target_per_location // 3 + 35,
        ),
    )

    for _ in range(max_rounds):
        if job.stop_event.is_set():
            break

        while job.pause_event.is_set():
            job.state = "paused"
            job.message = "متوقف مؤقتًا — اضغط استكمال"

            if job.stop_event.is_set():
                return collected

            await asyncio.sleep(0.4)

        job.state = "running"

        # ----------------------------------------------------
        # Preferred: actual result cards
        # ----------------------------------------------------
        articles = page.locator(
            'div[role="article"]'
        )

        article_count = await articles.count()

        for i in range(article_count):
            if len(collected) >= job.target_per_location:
                break

            try:
                article = articles.nth(i)

                href = ""

                anchors = article.locator(
                    'a[href*="/maps/place/"]'
                )

                if await anchors.count():
                    href = (
                        await anchors.first.get_attribute("href")
                        or ""
                    )

                href = clean_text(href)

                if href and not href.startswith("http"):
                    href = urljoin(
                        "https://www.google.com",
                        href,
                    )

                if not href:
                    # Some versions expose a click target rather than
                    # a conventional anchor.
                    try:
                        raw_text = clean_text(
                            await article.inner_text(timeout=1500)
                        )
                    except Exception:
                        raw_text = ""

                    if not raw_text:
                        continue

                    # No stable URL means we cannot safely dedupe later,
                    # so skip it for now.
                    continue

                key = href.split("?")[0]

                if key in job.seen_link_keys:
                    continue

                name = clean_text(
                    await article.get_attribute("aria-label")
                )

                if not name and await anchors.count():
                    name = clean_text(
                        await anchors.first.get_attribute("aria-label")
                    )

                try:
                    card_text = clean_text(
                        await article.inner_text(timeout=2000)
                    )
                except Exception:
                    card_text = ""

                if not name:
                    name = (
                        card_text.split("\n", 1)[0].strip()
                        if card_text
                        else ""
                    )

                job.seen_link_keys.add(key)

                item = {
                    "name": name,
                    "url": href,
                    "search_location": search_location,
                    "card_text": card_text,
                    "card_phone": parse_phone(card_text),
                    "card_rating": parse_rating(card_text),
                    "card_reviews": parse_reviews(card_text),
                }

                collected.append(item)

            except Exception:
                continue

        # ----------------------------------------------------
        # Fallback: scan all place links currently in DOM
        # ----------------------------------------------------
        links = page.locator(
            'a[href*="/maps/place/"]'
        )

        link_count = await links.count()

        for i in range(link_count):
            if len(collected) >= job.target_per_location:
                break

            try:
                link = links.nth(i)

                href = (
                    await link.get_attribute("href")
                    or ""
                )

                href = clean_text(href)

                if not href:
                    continue

                if not href.startswith("http"):
                    href = urljoin(
                        "https://www.google.com",
                        href,
                    )

                key = href.split("?")[0]

                if key in job.seen_link_keys:
                    continue

                name = clean_text(
                    await link.get_attribute("aria-label")
                )

                try:
                    link_text = clean_text(
                        await link.inner_text(timeout=1500)
                    )
                except Exception:
                    link_text = ""

                if not name:
                    name = link_text

                job.seen_link_keys.add(key)

                collected.append(
                    {
                        "name": name,
                        "url": href,
                        "search_location": search_location,
                        "card_text": link_text,
                        "card_phone": parse_phone(link_text),
                        "card_rating": parse_rating(link_text),
                        "card_reviews": parse_reviews(link_text),
                    }
                )

            except Exception:
                continue

        # Per-location result count; the parent worker aggregates these.
        loc_state = job.location_status.setdefault(
            search_location,
            {
                "state": "running",
                "found": 0,
                "processed": 0,
                "target": job.target_per_location,
                "message": "",
            },
        )
        loc_state["found"] = min(
            len(collected),
            job.target_per_location,
        )
        loc_state["message"] = (
            f"تم العثور على {loc_state['found']} نتيجة"
        )

        if len(collected) >= job.target_per_location:
            break

        # ----------------------------------------------------
        # Scroll the result list
        # ----------------------------------------------------
        feed = page.locator(
            'div[role="feed"]'
        ).first

        try:
            if await feed.count():
                await feed.evaluate(
                    "(el) => { el.scrollTop = el.scrollHeight; }"
                )
            else:
                await page.mouse.wheel(0, 4500)
        except Exception:
            try:
                await page.mouse.wheel(0, 4500)
            except Exception:
                pass

        await page.wait_for_timeout(1800)

        current_count = await page.locator(
            'div[role="article"]'
        ).count()

        current_links = await page.locator(
            'a[href*="/maps/place/"]'
        ).count()

        visible_count = max(
            current_count,
            current_links,
        )

        if visible_count <= previous_count:
            stable_rounds += 1
        else:
            stable_rounds = 0

        previous_count = visible_count

        # Don't exit too quickly: Google Maps may be loading
        # the next batch after an initially stable DOM.
        if stable_rounds >= 6:
            break

        job.message = (
            f"تم العثور على {len(job.link_items)} نتيجة. "
            f"جاري تحميل المزيد..."
        )

    return collected[:job.target_per_location]


# ============================================================
# Place details
# ============================================================

async def extract_place_details(
    context: BrowserContext,
    item: dict,
    category_hint: str,
) -> dict:
    page = await context.new_page()

    try:
        page.set_default_timeout(8000)

        await page.goto(
            item["url"],
            wait_until="domcontentloaded",
            timeout=45000,
        )

        # Details pages are also sometimes slow.
        await page.wait_for_timeout(1600)

        await accept_google_dialogs(page)

        name = await first_text(
            page,
            [
                "h1",
                'h1[role="heading"]',
            ],
        )

        if not name:
            name = clean_text(
                item.get("name", "")
            )

        address = await first_text(
            page,
            [
                '[data-item-id="address"]',
                'button[data-item-id="address"]',
                '[aria-label*="العنوان"]',
                '[aria-label*="Address"]',
            ],
        )

        phone = await find_phone(
            page,
            fallback_text=item.get("card_text", ""),
        )

        if not phone:
            phone = clean_text(
                item.get("card_phone", "")
            )

        website = await first_attr(
            page,
            [
                '[data-item-id="authority"] a',
                'a[data-item-id="authority"]',
                '[aria-label*="موقع إلكتروني"]',
                '[aria-label*="Website"]',
            ],
            "href",
        )

        category = await first_text(
            page,
            [
                '[data-item-id="category"]',
                'button[jsaction*="category"]',
            ],
        )

        if not category:
            category = category_hint

        try:
            body_text = clean_text(
                await page.locator("body").inner_text(
                    timeout=8000
                )
            )
        except Exception:
            body_text = ""

        if not phone:
            phone = parse_phone(
                item.get("card_text", "")
            )

        rating = (
            parse_rating(body_text)
            if parse_rating(body_text) is not None
            else item.get("card_rating")
        )

        reviews = (
            parse_reviews(body_text)
            if parse_reviews(body_text) is not None
            else item.get("card_reviews")
        )

        # Prefer current URL after any redirect.
        maps_url = page.url or item["url"]

        result = {
            "id": normalize_key(
                maps_url
                or name
            ),
            "business_name": name,
            "category": category,
            "phone": clean_text(phone),
            "rating": rating,
            "reviews": reviews,
            "address": address,
            "website": website,
            "maps_url": maps_url,
            "status": "",
        }

        return result

    finally:
        try:
            await page.close()
        except Exception:
            pass


# ============================================================
# Excel export
# ============================================================

def save_excel(job: ScrapeJob) -> str:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    location_slug = "_".join(
        safe_filename(x) for x in job.locations[:5]
    )

    filename = (
        f"leads_{safe_filename(job.category)}_"
        f"{location_slug or 'locations'}_"
        f"{timestamp}.xlsx"
    )

    output_path = OUTPUT_DIR / filename

    rows = []

    for item in job.results:
        rows.append(
            {
                "مكان البحث": item.get(
                    "search_location",
                    "",
                ),
                "اسم النشاط التجاري": item.get(
                    "business_name",
                    "",
                ),
                "نوع النشاط / التصنيف": item.get(
                    "category",
                    "",
                ),
                "رقم الهاتف": item.get(
                    "phone",
                    "",
                ),
                "التقييم": item.get(
                    "rating",
                    "",
                ),
                "عدد التقييمات": item.get(
                    "reviews",
                    "",
                ),
                "العنوان / المنطقة": item.get(
                    "address",
                    "",
                ),
                "الموقع الإلكتروني": item.get(
                    "website",
                    "",
                ),
                "رابط Google Maps": item.get(
                    "maps_url",
                    "",
                ),
                "حالة التواصل": item.get(
                    "status",
                    "",
                ),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty and "رابط Google Maps" in df.columns:
        df.drop_duplicates(
            subset=["رابط Google Maps"],
            inplace=True,
        )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Leads",
        )

        ws = writer.sheets["Leads"]

        ws.freeze_panes = "A2"

        if ws.max_row >= 1 and ws.max_column >= 1:
            ws.auto_filter.ref = ws.dimensions

        for column in ws.columns:
            max_length = 0
            letter = column[0].column_letter

            for cell in column:
                value = str(cell.value or "")
                max_length = max(
                    max_length,
                    len(value),
                )

            ws.column_dimensions[
                letter
            ].width = min(
                max(
                    12,
                    max_length + 2,
                ),
                50,
            )

    return str(output_path)


# ============================================================
# Worker
# ============================================================

async def run_scrape(job: ScrapeJob):
    global active_job_id

    job.state = "running"
    job.error = None

    # Initialize per-location state.
    for location in job.locations:
        job.location_status[location] = {
            "state": "queued",
            "found": 0,
            "processed": 0,
            "target": job.target_per_location,
            "message": "في قائمة الانتظار",
        }

    async with async_playwright() as p:
        browser = None

        try:
            launch_options = {
                "headless": HEADLESS,
                "args": [
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            }

            if BROWSER_CHANNEL:
                try:
                    browser = await p.chromium.launch(
                        channel=BROWSER_CHANNEL,
                        **launch_options,
                    )
                except Exception:
                    browser = await p.chromium.launch(
                        **launch_options,
                    )
            else:
                browser = await p.chromium.launch(
                    **launch_options,
                )

            job.browser = browser

            context = await browser.new_context(
                viewport={
                    "width": 1440,
                    "height": 1000,
                },
                locale="ar-EG",
                timezone_id="Africa/Cairo",
            )

            job.context = context

            # ------------------------------------------------
            # Phase 1: search multiple locations concurrently
            # ------------------------------------------------
            search_sem = asyncio.Semaphore(
                max(1, min(MAX_LOCATION_CONCURRENCY, len(job.locations)))
            )

            async def collect_one(location: str):
                if job.stop_event.is_set():
                    return []

                while job.pause_event.is_set():
                    job.state = "paused"

                    if job.stop_event.is_set():
                        return []

                    await asyncio.sleep(0.4)

                job.location_status[location]["state"] = "running"
                job.location_status[location]["message"] = (
                    "جاري البحث..."
                )

                async with search_sem:
                    page = await context.new_page()

                    try:
                        page.set_default_timeout(12000)

                        page.on(
                            "pageerror",
                            lambda exc: print(
                                "[LeadScout pageerror]",
                                str(exc)[:800],
                            ),
                        )

                        results = await open_search_and_collect(
                            job,
                            page,
                            location,
                        )

                        job.location_status[location]["found"] = len(results)
                        job.location_status[location]["state"] = "collected"
                        job.location_status[location]["message"] = (
                            f"تم جمع {len(results)} نتيجة"
                        )
                        job.found = sum(
                            int(v.get("found", 0))
                            for v in job.location_status.values()
                        )

                        return results

                    except asyncio.CancelledError:
                        raise

                    except Exception as exc:
                        job.location_status[location]["state"] = "error"
                        job.location_status[location]["message"] = (
                            f"خطأ: {str(exc)[:250]}"
                        )
                        job.error = str(exc)[:2000]
                        print(
                            "[LeadScout search error]",
                            location,
                            str(exc)[:800],
                        )
                        return []

                    finally:
                        try:
                            await page.close()
                        except Exception:
                            pass

            collected_groups = await asyncio.gather(
                *[
                    collect_one(location)
                    for location in job.locations
                ],
                return_exceptions=True,
            )

            if job.stop_event.is_set():
                job.state = "stopped"
                job.message = "تم إيقاف العملية"
                return

            # Flatten and globally deduplicate by Maps URL.
            all_links: list[dict] = []

            for group in collected_groups:
                if isinstance(group, Exception):
                    continue

                all_links.extend(group)

            unique_links: list[dict] = []

            for item in all_links:
                key = normalize_key(
                    item.get("url", "")
                )

                if not key:
                    continue

                if key in job.seen_link_keys:
                    continue

                job.seen_link_keys.add(key)
                unique_links.append(item)

            # Re-apply the per-location caps after global de-duplication.
            per_location_counts: dict[str, int] = {}
            final_links: list[dict] = []

            for item in unique_links:
                location = item.get(
                    "search_location",
                    "",
                )

                count = per_location_counts.get(
                    location,
                    0,
                )

                if count >= job.target_per_location:
                    continue

                per_location_counts[location] = count + 1
                final_links.append(item)

            job.link_items = final_links
            job.found = len(final_links)

            for location in job.locations:
                job.location_status[location]["found"] = (
                    per_location_counts.get(location, 0)
                )
                job.location_status[location]["state"] = "ready"
                job.location_status[location]["message"] = (
                    f"تم تجهيز {job.location_status[location]['found']} نتيجة"
                )

            if not final_links:
                raise RuntimeError(
                    "لم يتم العثور على روابط نتائج من أي موقع."
                )

            # ------------------------------------------------
            # Phase 2: concurrent detail extraction
            # ------------------------------------------------
            job.message = (
                f"تم جمع {len(final_links)} نتيجة من "
                f"{len(job.locations)} موقع — جاري قراءة التفاصيل..."
            )

            detail_sem = asyncio.Semaphore(
                max(1, DETAIL_CONCURRENCY)
            )

            async def process_one(index: int, item: dict):
                if job.stop_event.is_set():
                    return

                while job.pause_event.is_set():
                    job.state = "paused"

                    if job.stop_event.is_set():
                        return

                    await asyncio.sleep(0.4)

                job.state = "running"

                location = item.get(
                    "search_location",
                    "",
                )

                async with detail_sem:
                    try:
                        result = await extract_place_details(
                            context,
                            item,
                            job.category,
                        )

                        result["search_location"] = location

                        dedupe_key = normalize_key(
                            result.get("maps_url")
                            or result.get("business_name")
                        )

                        if (
                            dedupe_key
                            and dedupe_key
                            not in job.seen_result_keys
                        ):
                            job.seen_result_keys.add(
                                dedupe_key
                            )
                            job.results.append(result)

                    except asyncio.CancelledError:
                        raise

                    except Exception as exc:
                        print(
                            "[LeadScout detail error]",
                            item.get("name", ""),
                            "location=",
                            location,
                            str(exc)[:500],
                        )

                    finally:
                        job.processed += 1

                        loc_state = job.location_status.get(
                            location
                        )

                        if loc_state:
                            loc_state["processed"] += 1
                            loc_state["state"] = "processing"

                            if (
                                loc_state["processed"]
                                >= loc_state["found"]
                            ):
                                loc_state["state"] = "finished"

                        job.message = (
                            f"جاري استخراج التفاصيل: "
                            f"{job.processed}/{len(final_links)}"
                        )

            # Process in controlled parallel batches through the semaphore.
            tasks = [
                asyncio.create_task(
                    process_one(index, item)
                )
                for index, item in enumerate(
                    final_links,
                    start=1,
                )
            ]

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            if job.stop_event.is_set():
                job.state = "stopped"
                job.message = "تم إيقاف العملية"
                return

            job.results = job.results[:]

            # Anything with no explicit location is marked from the original item.
            for result in job.results:
                if not result.get("search_location"):
                    result["search_location"] = ""

            job.message = "جاري إنشاء ملف Excel..."

            job.output_file = save_excel(job)

            job.state = "finished"

            job.message = (
                f"اكتمل الاستخراج — "
                f"تم حفظ {len(job.results)} نتيجة"
            )

        except asyncio.CancelledError:
            job.state = "stopped"
            job.message = "تم إلغاء العملية"

        except Exception as exc:
            job.state = "error"
            job.error = str(exc)[:2000]
            job.message = (
                f"حدث خطأ: {job.error}"
            )

        finally:
            try:
                if job.context:
                    await job.context.close()
            except Exception:
                pass

            try:
                if browser:
                    await browser.close()
            except Exception:
                pass

            job.browser = None
            job.context = None
            job.page = None

            if active_job_id == job.job_id:
                active_job_id = None


# ============================================================
# API
# ============================================================

@app.get(
    "/",
    include_in_schema=False,
)
async def root():
    index_file = WEB_DIR / "index.html"

    if index_file.exists():
        return FileResponse(
            index_file,
            media_type="text/html; charset=utf-8",
        )

    return {
        "app": APP_NAME,
        "status": "online",
        "message": "ضع index.html داخل مجلد web/",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "browser": BROWSER_CHANNEL,
        "headless": HEADLESS,
        "time": datetime.now().isoformat(
            timespec="seconds"
        ),
    }


@app.post("/api/start-scrape")
async def start_scrape(
    request: ScrapeRequest,
):
    global active_job_id

    if active_job_id:
        existing = jobs.get(
            active_job_id
        )

        if existing and existing.state in {
            "running",
            "paused",
        }:
            raise HTTPException(
                status_code=409,
                detail="هناك عملية استخراج تعمل بالفعل.",
            )

    job_id = uuid.uuid4().hex

    locations: list[str] = []

    if request.locations:
        locations.extend(
            clean_text(x)
            for x in request.locations
            if clean_text(x)
        )

    if request.location and clean_text(request.location):
        locations.append(
            clean_text(request.location)
        )

    # Deduplicate locations while preserving order.
    unique_locations: list[str] = []
    seen_locations: set[str] = set()

    for location in locations:
        key = normalize_key(location)

        if key and key not in seen_locations:
            seen_locations.add(key)
            unique_locations.append(location)

    if not unique_locations:
        raise HTTPException(
            status_code=422,
            detail="أضف موقعًا واحدًا على الأقل.",
        )

    job = ScrapeJob(
        job_id=job_id,
        category=request.category.strip(),
        locations=unique_locations,
        target_per_location=request.max_results,
    )

    job.pause_event.clear()
    job.stop_event.clear()

    jobs[job_id] = job
    active_job_id = job_id

    job.task = asyncio.create_task(
        run_scrape(job)
    )

    return {
        "status": "started",
        "job_id": job_id,
        "message": "تم بدء عملية الاستخراج",
        "status_url": (
            f"/api/status/{job_id}"
        ),
        "download_url": (
            f"/api/download/{job_id}"
        ),
    }


@app.get("/api/status/{job_id}")
async def get_status(
    job_id: str,
):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    total_target = (
        job.target_per_location
        * len(job.locations)
    )

    progress = 0.0

    if total_target > 0:
        progress = round(
            min(
                job.found / total_target,
                1.0,
            ) * 100,
            1,
        )

    return {
        "job_id": job.job_id,
        "state": job.state,
        "message": job.message,
        "category": job.category,
        "locations": job.locations,
        "target_per_location": job.target_per_location,
        "target": total_target,
        "found": job.found,
        "processed": job.processed,
        "progress": progress,
        "locations_status": job.location_status,
        "error": job.error,
        "has_file": bool(
            job.output_file
            and Path(
                job.output_file
            ).exists()
        ),
        "download_url": (
            f"/api/download/{job.job_id}"
            if job.output_file
            else None
        ),
    }


@app.post("/api/pause/{job_id}")
async def pause_scrape(
    job_id: str,
):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    if job.state != "running":
        return {
            "status": "ignored",
            "message": (
                f"الحالة الحالية: {job.state}"
            ),
        }

    job.pause_event.set()
    job.state = "paused"
    job.message = (
        "متوقف مؤقتًا — اضغط استكمال"
    )

    return {
        "status": "paused",
        "job_id": job_id,
    }


@app.post("/api/resume/{job_id}")
async def resume_scrape(
    job_id: str,
):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    if job.state != "paused":
        return {
            "status": "ignored",
            "message": (
                f"الحالة الحالية: {job.state}"
            ),
        }

    job.pause_event.clear()
    job.state = "running"
    job.message = "جاري الاستكمال..."

    return {
        "status": "resumed",
        "job_id": job_id,
    }


@app.post("/api/stop/{job_id}")
async def stop_scrape(
    job_id: str,
):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    job.stop_event.set()
    job.pause_event.clear()

    if job.state in {
        "running",
        "paused",
    }:
        job.state = "stopped"

    job.message = "تم إيقاف العملية"

    return {
        "status": "stopped",
        "job_id": job_id,
    }


@app.get("/api/results/{job_id}")
async def get_results(
    job_id: str,
):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return {
        "job_id": job.job_id,
        "count": len(job.results),
        "results": job.results,
    }


@app.get("/api/download/{job_id}")
async def download_file(
    job_id: str,
):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    if not job.output_file:
        raise HTTPException(
            status_code=404,
            detail="الملف غير جاهز بعد.",
        )

    path = Path(job.output_file)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="الملف غير موجود.",
        )

    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
    )


@app.get("/api/jobs")
async def list_jobs():
    return {
        "count": len(jobs),
        "jobs": [
            {
                "job_id": job.job_id,
                "category": job.category,
                "locations": job.locations,
                "state": job.state,
                "found": job.found,
                "target_per_location": job.target_per_location,
                "target": job.target_per_location * len(job.locations),
                "message": job.message,
            }
            for job in jobs.values()
        ],
    }


# ============================================================
# Local entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
