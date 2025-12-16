import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# -------------------- CONFIG --------------------
BASE_URL = "https://www.olx.uz/d/nedvizhimost/kvartiry/arenda-dolgosrochnaya/tashkent/"
MAX_PAGES = 20
PAGE_RETRY = 2
WAIT_TIMEOUT = 12
SCROLL_PAUSE = 1
OUTPUT_CSV = "olx_rentals_tashkent_full.csv"
# ------------------------------------------------

# Desktop user-agent
desktop_ua = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def make_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"user-agent={desktop_ua}")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_window_size(1920, 1080)
    return driver


def scroll_to_bottom(driver, pause=SCROLL_PAUSE, max_scroll_loops=30):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scroll_loops):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


# -------------------- EXTRACT OVERVIEW ADS --------------------

def extract_price_and_currency(text):
    """Extract numeric price and currency from text"""
    if not text:
        return None, None

    # Detect currency
    currency = None
    if "у.е." in text or "у. е." in text:
        currency = "у.е."
    elif "сум" in text or "сўм" in text:
        currency = "сум"
    elif "$" in text:
        currency = "$"

    # Extract numeric value
    txt = text.replace("сум", "").replace("сўм", "").replace("у.е.", "").replace("у. е.", "").replace("$", "")
    txt = txt.replace(",", "").replace(" ", "").strip()
    numeric = ''.join(ch for ch in txt if ch.isdigit())

    return (numeric if numeric else None), currency


def extract_rooms_from_text(text):
    """Extract number of rooms from various text formats"""
    if not text:
        return None

    text_lower = text.lower()

    # Pattern 1: "2-комнатная", "3-хонали", "1-х комнатная"
    match = re.search(r'(\d+)[-\s]*(?:x|х)?(?:онали|комн|ком|xona)', text_lower)
    if match:
        return match.group(1)

    # Pattern 2: "2 комнаты", "3 комната"
    match = re.search(r'(\d+)\s+комн', text_lower)
    if match:
        return match.group(1)

    # Pattern 3: Look for standalone digit before keywords
    match = re.search(r'(\d)\s*(?:ком|xona)', text_lower)
    if match:
        return match.group(1)

    return None


def extract_ads_from_html(html):
    """Extract ad listings from the main page HTML"""
    soup = BeautifulSoup(html, "html.parser")

    # Find all ad links
    ad_links = soup.find_all("a", href=True)

    listings = []
    seen_urls = set()

    for link in ad_links:
        href = link.get("href", "")

        # Only process rental ad links
        if "/d/obyavlenie/" not in href:
            continue

        # Make full URL
        url = href if href.startswith("http") else "https://www.olx.uz" + href

        # Skip duplicates
        if url in seen_urls:
            continue
        seen_urls.add(url)

        try:
            # Extract title - look for h4 or h6 within the link
            title_el = link.find("h4") or link.find("h6")
            title = title_el.get_text(strip=True) if title_el else None

            # Skip if title is generic/navigation text
            if title and title.lower() in ['уведомления', 'чат', 'ваш профиль', 'подать объявление']:
                continue

            # Extract rooms from title
            rooms = extract_rooms_from_text(title) if title else None

            # Find parent container
            parent = link.find_parent("div", recursive=True)

            # Extract price and currency
            price_text = None
            if parent:
                price_elements = parent.find_all(
                    string=lambda text: text and ("у.е." in text or "сум" in text or "сўм" in text or "$" in text))
                if price_elements:
                    price_text = price_elements[0].strip()

            price, currency = extract_price_and_currency(price_text)

            # Extract district
            district = None
            if parent:
                location_elements = parent.find_all(string=lambda text: text and "район" in text.lower())
                if location_elements:
                    loc_text = location_elements[0].strip()
                    district = loc_text.split(",")[0].split("-")[0].strip()

            # Extract area (m²)
            area = None
            if parent:
                # Look for text with m² or м²
                area_elements = parent.find_all(string=lambda text: text and ("m²" in text or "м²" in text))
                if area_elements:
                    area_text = area_elements[0].strip()
                    area_match = re.search(r'(\d+)\s*(?:m²|м²)', area_text)
                    if area_match:
                        area = area_match.group(1)

            listings.append({
                "title": title,
                "price": price,
                "currency": currency,
                "district": district,
                "subdistrict": None,  # Will be filled from detail page
                "rooms": rooms,
                "area_m2": area,
                "floor": None,
                "total_floors": None,
                "furnished": None,
                "property_type": None,
                "posted_date": None,
                "url": url
            })

        except Exception as e:
            print(f"Error extracting ad: {e}")
            continue

    return listings


# -------------------- EXTRACT DETAILED AD DATA --------------------

def extract_details_from_ad_page(driver, url):
    """Extract detailed information from individual ad page"""
    try:
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        details = {
            "district": None,
            "subdistrict": None,
            "rooms": None,
            "area_m2": None,
            "floor": None,
            "total_floors": None,
            "furnished": None,
            "property_type": None,
            "posted_date": None,
            "title": None,
            "price": None,
            "currency": None
        }

        # Extract title - look for h1 first, then h4
        title_el = soup.find("h1")
        if not title_el:
            title_el = soup.find("h4")
        if title_el:
            title_text = title_el.get_text(strip=True)
            # Skip generic titles
            if title_text.lower() not in ['уведомления', 'чат', 'ваш профиль']:
                details["title"] = title_text
                # Extract rooms from title
                details["rooms"] = extract_rooms_from_text(title_text)

        # Extract price and currency
        price_elements = soup.find_all(
            string=lambda text: text and ("у.е." in text or "сум" in text or "сўм" in text or "$" in text))
        if price_elements:
            for price_el in price_elements:
                price_text = price_el.strip()
                if len(price_text) < 50 and any(char.isdigit() for char in price_text):
                    price, currency = extract_price_and_currency(price_text)
                    if price:
                        details["price"] = price
                        details["currency"] = currency
                        break

        # Look for structured parameters (dl/dt/dd tags)
        param_lists = soup.find_all('dl')
        for dl in param_lists:
            dt_elements = dl.find_all('dt')
            dd_elements = dl.find_all('dd')

            for dt, dd in zip(dt_elements, dd_elements):
                label = dt.get_text(strip=True).lower()
                value = dd.get_text(strip=True)

                # District/Subdistrict
                if 'район' in label or 'district' in label:
                    if 'subdistrict' in label or 'микрорайон' in label:
                        details["subdistrict"] = value
                    else:
                        details["district"] = value

                # Rooms
                elif 'комнат' in label or 'room' in label or 'xona' in label:
                    room_match = re.search(r'(\d+)', value)
                    if room_match:
                        details["rooms"] = room_match.group(1)
                    else:
                        details["rooms"] = extract_rooms_from_text(value)

                # Area
                elif 'площад' in label or 'area' in label or 'maydon' in label:
                    area_match = re.search(r'(\d+)', value)
                    if area_match:
                        details["area_m2"] = area_match.group(1)

                # Floor
                elif 'этаж' in label or 'floor' in label or 'qavat' in label:
                    if '/' in value:
                        parts = value.split('/')
                        if len(parts) == 2:
                            details["floor"] = parts[0].strip()
                            details["total_floors"] = parts[1].strip()
                    else:
                        floor_match = re.search(r'(\d+)', value)
                        if floor_match:
                            details["floor"] = floor_match.group(1)

                # Furnished
                elif 'мебел' in label or 'furnish' in label or 'mebel' in label:
                    details["furnished"] = value

                # Property type
                elif 'тип' in label or 'type' in label:
                    details["property_type"] = value

        # If structured params not found, use regex on full text
        full_text = soup.get_text()

        # Extract location info from text
        if not details["district"]:
            location_match = re.search(r'([\w\-]+ский район)', full_text)
            if location_match:
                details["district"] = location_match.group(1)

        # Try to find subdistrict from location text
        # Pattern: "Ташкент, Район, Массив/Квартал"
        location_lines = [line.strip() for line in full_text.split('\n') if
                          'район' in line.lower() and len(line.strip()) < 100]
        for line in location_lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                # Usually format: City, District, Subdistrict
                if 'район' in parts[1].lower():
                    details["district"] = parts[1]
                    if len(parts) > 2:
                        details["subdistrict"] = parts[2]
                    break

        # Extract rooms from text if not found
        if not details["rooms"]:
            room_patterns = [
                r'(\d+)[-\s]*(?:комнатн|комн|xona|х)',
                r'(\d+)\s+комн',
            ]
            for pattern in room_patterns:
                room_match = re.search(pattern, full_text.lower())
                if room_match:
                    details["rooms"] = room_match.group(1)
                    break

        # Extract area if not found
        if not details["area_m2"]:
            area_match = re.search(r'(\d+)\s*(?:m²|м²|кв\.?\s*м)', full_text)
            if area_match:
                details["area_m2"] = area_match.group(1)

        # Extract floor info if not found
        if not details["floor"]:
            floor_match = re.search(r'(\d+)\s*/\s*(\d+)', full_text)
            if floor_match:
                details["floor"] = floor_match.group(1)
                details["total_floors"] = floor_match.group(2)

        # Extract posted date
        date_patterns = [
            r'(Сегодня|Вчера)',
            r'(\d{1,2}\s+\w+\s+\d{4})',
            r'(\d{1,2}\s+\w+)',
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, full_text)
            if date_match:
                details["posted_date"] = date_match.group(1)
                break

        return details

    except Exception as e:
        print(f"Error extracting details from {url}: {e}")
        return {
            "district": None, "subdistrict": None, "rooms": None,
            "area_m2": None, "floor": None, "total_floors": None,
            "furnished": None, "property_type": None, "posted_date": None,
            "title": None, "price": None, "currency": None
        }


# -------------------- MAIN --------------------

def main():
    driver = make_driver()
    all_listings = []

    try:
        for page in range(1, MAX_PAGES + 1):
            page_url = BASE_URL if page == 1 else f"{BASE_URL}?page={page}"
            print(f"\n🔎 Loading page {page}: {page_url}")
            driver.get(page_url)

            try:
                WebDriverWait(driver, WAIT_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "a"))
                )
            except:
                print("⚠️ Timeout waiting for page to load")
                pass

            scroll_to_bottom(driver)
            time.sleep(2)

            html = driver.page_source
            listings = extract_ads_from_html(html)
            print(f"✅ Found {len(listings)} ads on page {page}")

            if listings:
                print(f"Sample ad: {listings[0]}")

            all_listings.extend(listings)
            time.sleep(2)

    finally:
        driver.quit()

    if not all_listings:
        print("❌ No ads scraped - check selectors!")
        return

    print(f"\n📊 Total ads collected from listing pages: {len(all_listings)}")
    df = pd.DataFrame(all_listings)

    # -------- Enrich with detail page data --------
    print("\n🔍 Starting detailed scraping...")
    driver = make_driver()
    try:
        for i, row in df.iterrows():
            url = row["url"]
            if pd.isna(url):
                continue

            print(f"🔍 Processing ad {i + 1}/{len(df)}: {url}")
            details = extract_details_from_ad_page(driver, url)

            # Update row with details (only if current value is null/empty)
            for k, v in details.items():
                if k in df.columns:
                    current_val = df.at[i, k]
                    if (pd.isna(current_val) or current_val == "" or current_val is None) and v is not None and v != "":
                        df.at[i, k] = v

            # Show progress with key fields
            print(f"   Title: {df.at[i, 'title'][:50] if pd.notna(df.at[i, 'title']) else 'N/A'}...")
            print(
                f"   Rooms: {df.at[i, 'rooms']}, District: {df.at[i, 'district']}, Subdistrict: {df.at[i, 'subdistrict']}")

            time.sleep(1.5)

    finally:
        driver.quit()

    # Convert price to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Remove rows without price
    initial_count = len(df)
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    removed = initial_count - len(df)
    if removed > 0:
        print(f"\n🗑️ Removed {removed} ads without valid price")

    # Reorder columns for better readability
    column_order = [
        "title", "price", "currency", "rooms", "area_m2",
        "district", "subdistrict", "floor", "total_floors",
        "furnished", "property_type", "posted_date", "url"
    ]
    df = df[[col for col in column_order if col in df.columns]]

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved {len(df)} ads to: {OUTPUT_CSV}")

    print("\n📋 Preview of data:")
    print(df.head(10).to_string())

    print("\n📊 Data Quality Summary:")
    print(f"Total records: {len(df)}")
    print(f"With rooms: {df['rooms'].notna().sum()} ({df['rooms'].notna().sum() / len(df) * 100:.1f}%)")
    print(f"With district: {df['district'].notna().sum()} ({df['district'].notna().sum() / len(df) * 100:.1f}%)")
    print(
        f"With subdistrict: {df['subdistrict'].notna().sum()} ({df['subdistrict'].notna().sum() / len(df) * 100:.1f}%)")
    print(f"With area: {df['area_m2'].notna().sum()} ({df['area_m2'].notna().sum() / len(df) * 100:.1f}%)")
    print(f"With floor: {df['floor'].notna().sum()} ({df['floor'].notna().sum() / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()