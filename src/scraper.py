import os
import json
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

# --- Configuration ---
BASE_URL = "https://academic-calendar.wlu.ca"
OUTPUT_DIR = os.path.join("data", "raw")
MAX_DOCS = 500  # Comfortably within the 200-2,000 requirement

# WLU specific targeting
ALLOWED_SCRIPTS = {"index_old.php", "section.php", "department.php", "course.php"}
STOP_MARKERS = ["Academic & Related Dates", "Other Calendars", "Contact WLU", "Wilfrid Laurier University | 75 University Avenue"]
NOISE_RE = re.compile(r"(nav|menu|sidebar|footer|header|search|breadcrumb|toolbar)", re.I)

def canonical_key(url):
    """Creates a unique identifier for a page to avoid downloading duplicates."""
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return None
    script = parsed.path.rsplit("/", 1)[-1]
    if script not in ALLOWED_SCRIPTS:
        return None
    
    qs = parse_qs(parsed.query)
    parts = [script]
    for key in ["cal", "y", "s", "ss", "sp", "d", "c"]:
        if key in qs:
            parts.append(f"{key}={qs[key][0]}")
    return "|".join(parts)

def extract_content(html):
    """Cleans the HTML and extracts just the title and main text."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract Title
    h1 = soup.find(["h1", "h2"])
    title = h1.get_text(strip=True) if h1 else (soup.title.string if soup.title else "Untitled")
    
    # Strip useless web elements (nav bars, footers)
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "title"]):
        tag.decompose()
    for el in soup.find_all(id=NOISE_RE): el.decompose()
    for el in soup.find_all(class_=NOISE_RE): el.decompose()

    # Extract Text
    lines = [line.strip() for line in soup.get_text(separator="\n").split("\n") if line.strip()]
    
    # Cut off text after footers or sidebars
    for i, line in enumerate(lines):
        if any(marker.lower() in line.lower() for marker in STOP_MARKERS):
            lines = lines[:i]
            break

    return {"title": title, "text": "\n".join(lines).strip()}

def scrape():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start at the Undergraduate Calendar (cal=1, year=93 is standard)
    start_url = f"{BASE_URL}/index_old.php?{urlencode({'cal': '1', 'y': '93'})}"
    
    queue = [start_url]
    visited_keys = set()
    docs_saved = 0

    print(f"Starting crawl at {start_url}...")

    while queue and docs_saved < MAX_DOCS:
        current_url = queue.pop(0)
        c_key = canonical_key(current_url)
        
        if not c_key or c_key in visited_keys:
            continue
            
        visited_keys.add(c_key)
        time.sleep(0.5) # Polite delay

        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200: continue
            
            # Save the document if it's a content page (not just the index router)
            script = urlparse(current_url).path.rsplit("/", 1)[-1]
            if script != "index_old.php":
                content = extract_content(response.text)
                
                # Only save if it actually grabbed substantial text
                if len(content["text"].split()) > 20:
                    doc_id = f"doc_{docs_saved:03d}"
                    doc_data = {
                        "doc_id": doc_id,
                        "url": current_url,
                        "title": content["title"],
                        "text": content["text"]
                    }
                    
                    with open(os.path.join(OUTPUT_DIR, f"{doc_id}.json"), "w", encoding="utf-8") as f:
                        json.dump(doc_data, f, indent=4, ensure_ascii=False)
                    
                    docs_saved += 1
                    print(f"[{docs_saved}/{MAX_DOCS}] Saved: {content['title']}")

            # Find new links to add to the queue
            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True):
                next_url = urljoin(current_url, a["href"]).split("#")[0]
                if canonical_key(next_url) not in visited_keys:
                    queue.append(next_url)

        except Exception as e:
            print(f"Failed to fetch {current_url}: {e}")

    print(f"\nFinished! Saved {docs_saved} documents to {OUTPUT_DIR}/")

if __name__ == "__main__":
    scrape()