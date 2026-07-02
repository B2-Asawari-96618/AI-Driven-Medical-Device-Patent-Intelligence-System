# Updated scraper template\n\nimport time
import json
import csv
import re
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
import re


from tqdm import tqdm
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
#from deep_translator import GoogleTranslator

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# =================================================
# DRIVER
# =================================================

def get_driver():

    options = Options()

    # options.add_argument("--headless=new")
    prefs = {
    "profile.managed_default_content_settings.images": 2
}

    options.add_experimental_option("prefs", prefs)

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-images")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)

    return driver



def extract_claims_from_google(patent_id,driver):

    url = f"https://patents.google.com/patent/{patent_id}/en"

    # headers = {
    #     "User-Agent":
    #     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    # }

    # r = requests.get(url, headers=headers, timeout=30)
    html = driver.page_source                                ### tushar

    soup = BeautifulSoup(
        html,
    "html.parser"
    )

    claims, count, indep = extract_claims(soup)

    # soup = BeautifulSoup(r.text, "html.parser")

    claims = []

    # New structure
    for div in soup.select("div.claim"):

        text = div.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)

        if len(text) > 20:
            claims.append(text)

    # Old structure fallback
    if not claims:

        for c in soup.find_all("claim"):

            text = c.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text)

            if len(text) > 20:
                claims.append(text)

    claims = list(dict.fromkeys(claims))

    independent_count = 0

    for claim in claims:

        if not re.search(
                r'(claim|claims)\s+\d+',
                claim.lower()):
            independent_count += 1

    return (
        "\n\n".join(claims),
        len(claims),
        independent_count
    )

# =================================================
# PATENT ID
# =================================================

def extract_patent_id(url):

    try:
        return url.split("/patent/")[1].split("/")[0]

    except:
        return None


# =================================================
# COUNTRY
# =================================================

def extract_country(patent_id):

    if patent_id:
        return patent_id[:2]

    return None





def extract_claims(soup):

    selectors = [
        "claim",
        "div[id^='CLM-']",
        "div.claim",
        "div.claim-text",
        "div[id^='c-en-']"
    ]

    all_claims = []
    seen = set()

    for selector in selectors:

        claim_tags = soup.select(selector)

        if claim_tags:

            print(
                f"Claims found using {selector}:",
                len(claim_tags)
            )

            for tag in claim_tags:

                text = tag.get_text(
                    " ",
                    strip=True
                )

                text = re.sub(
                    r"\s+",
                    " ",
                    text
                )

                if (
                    len(text) > 20 and
                    text not in seen
                ):

                    seen.add(text)
                    all_claims.append(text)

            break

    claim_count = len(all_claims)

    independent_count = 0

    for claim in all_claims:

        if not re.search(
            r'(claim|claims)\s+\d+',
            claim.lower()
        ):
            independent_count += 1

    return (
        "\n\n".join(all_claims),
        claim_count,
        independent_count
    )



def extract_description(soup):
    desc_selectors = [
        "section[itemprop='description']",
        "div.description",
        "div#description",
        "meta[name='description']",
        "div.abstract-description",
        ".description",
        "div.description-text"
    ]
    
    for selector in desc_selectors:
        try:
            if selector.startswith("meta"):
                el = soup.select_one(selector)
                if el and el.get("content"):
                    content = el.get("content").strip()
                    if len(content) > 50:
                        return content
            else:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text(" ", strip=True)
                    if text and len(text) > 50:
                        return text
        except:
            continue
    
    return None




def extract_ipc_cpc(driver, soup):
    """
    Extract IPC and CPC codes from Google Patents page.
    Uses multiple methods to find classification codes.
    """
    ipc = set()
    cpc = set()
    
    # ============================================
    # METHOD 1: Try Selenium with multiple selectors
    # ============================================
    classification_selectors = [
        "classification-tree span",
        ".classification",
        "span.classification",
        "div.cpc",
        "div.ipc",
        ".cpc-code",
        ".ipc-code",
        "td.classification",
        "span.cpc",
        "span.ipc",
        "[itemprop='classification']",
        ".classifications span",
        ".patent-classifications span"
    ]
    
    for selector in classification_selectors:
        try:
            elements = soup.select(selector)
            #elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements:
                txt = el.text.strip()
                if txt:
                    # Look for IPC/CPC patterns
                    matches = re.findall(r'[A-HY]\d{2}[A-Z]\d+/\d+', txt)
                    for m in matches:
                        ipc.add(m)
                        cpc.add(m)
            if ipc:
                print(f"Found {len(ipc)} codes using Selenium selector: {selector}")
                break
        except Exception as e:
            continue
    
    # ============================================
    # METHOD 2: Try BeautifulSoup if Selenium failed
    # ============================================
    if not ipc:
        soup_selectors = [
            ".classification",
            ".cpc",
            ".ipc",
            ".cpc-code",
            ".ipc-code",
            "td.classification",
            "span.classification",
            "div.classification",
            "[itemprop='classification']",
            ".classifications span",
            ".patent-classifications span",
            "classification-tree span"
        ]
        
        for selector in soup_selectors:
            try:
                elements = soup.select(selector)
                for el in elements:
                    txt = el.get_text(" ", strip=True)
                    matches = re.findall(r'[A-HY]\d{2}[A-Z]\d+/\d+', txt)
                    for m in matches:
                        ipc.add(m)
                        cpc.add(m)
                if ipc:
                    print(f"Found {len(ipc)} codes using BeautifulSoup selector: {selector}")
                    break
            except:
                continue
    
    # ============================================
    # METHOD 3: Search the entire page source (fallback)
    # ============================================
    if not ipc:
        try:
            page_source = driver.page_source
            # Look for IPC patterns in the entire page
            matches = re.findall(r'[A-HY]\d{2}[A-Z]\d+/\d+', page_source)
            for m in matches:
                # Filter out false positives (patent numbers, etc.)
                if len(m) >= 7 and not m.startswith('US') and not m.startswith('WO'):
                    ipc.add(m)
                    cpc.add(m)
            if ipc:
                print(f"Found {len(ipc)} codes from page source")
        except:
            pass
    
    # ============================================
    # METHOD 4: Look for classification in meta tags
    # ============================================
    if not ipc:
        try:
            meta_selectors = [
                "meta[name='citation_classification']",
                "meta[name='citation_ipc']",
                "meta[name='citation_cpc']"
            ]
            for selector in meta_selectors:
                elements = soup.select(selector)
                for el in elements:
                    content = el.get("content")
                    if content:
                        matches = re.findall(r'[A-HY]\d{2}[A-Z]\d+/\d+', content)
                        for m in matches:
                            ipc.add(m)
                            cpc.add(m)
                if ipc:
                    break
        except:
            pass
    
    # ============================================
    # METHOD 5: Look for classification in text
    # ============================================
    if not ipc:
        try:
            text = soup.get_text()
            # Look for common IPC patterns
            ipc_patterns = [
                r'(?i)ipc[:\s]+([A-HY]\d{2}[A-Z]\d+/\d+)',
                r'(?i)classification[:\s]+([A-HY]\d{2}[A-Z]\d+/\d+)',
                r'(?i)cpc[:\s]+([A-HY]\d{2}[A-Z]\d+/\d+)'
            ]
            for pattern in ipc_patterns:
                matches = re.findall(pattern, text)
                for m in matches:
                    ipc.add(m)
                    cpc.add(m)
                if ipc:
                    break
        except:
            pass
    
    return list(ipc), list(cpc)


# =================================================
# DATES
# =================================================
def extract_filing_date(soup):

    # Method 1
    try:
        filing = soup.select_one(
            "time[itemprop='filingDate']"
        )

        if filing:
            return filing.get("datetime")
    except:
        pass

    # Method 2: Application timeline
    try:

        events = soup.select(
            "div.event"
        )

        for event in events:

            text = event.get_text(
                " ",
                strip=True
            ).lower()

            if "application filed" in text:

                date_div = event.select_one(
                    "div.filed"
                )

                if date_div:
                    return date_div.get_text(
                        strip=True
                    )

    except Exception as e:
        print("Filing date error:", e)

    return None



def extract_publication_date(soup):
    # First check meta tag (most reliable for Google Patents)
    meta_pub = soup.select_one("meta[name='citation_publication_date']")
    if meta_pub and meta_pub.get("content"):
        return meta_pub.get("content").strip()
    
    # Then try time elements
    pub_selectors = [
        "time[itemprop='publicationDate']",
        "time.publication-date",
        "td.published",
        "div.publication",
        ".publication-date",
        ".date-published"
    ]
    
    for selector in pub_selectors:
        try:
            el = soup.select_one(selector)
            if el:
                date = el.get("datetime") or el.get_text(strip=True)
                if date and re.search(r'\d{4}', date):
                    return date.strip()
        except:
            continue
    
    # Finally try event timeline
    try:
        events = soup.select("div.event")
        for event in events:
            text = event.get_text(" ", strip=True).lower()
            if "publication" in text:
                date_div = event.select_one("div.publication, .date, time")
                if date_div:
                    date_text = date_div.get_text(strip=True)
                    if re.search(r'\d{4}', date_text):
                        return date_text
    except:
        pass
    
    return None


def extract_inventors(soup):
    inventors = set()
    
    # Primary selectors for inventor names
    inventor_selectors = [
        "[itemprop='inventor']",
        ".inventor",
        ".inventors a",
        "span.inventor",
        "div.inventors span",
        "a[itemprop='inventor']",
        ".inventors .name",
        "tr.inventor td"
    ]
    
    for selector in inventor_selectors:
        try:
            elements = soup.select(selector)
            for el in elements:
                txt = el.get_text(" ", strip=True)
                if txt and len(txt) > 2:
                    inventors.add(txt)
            if inventors:
                break
        except:
            continue
    
    # IMPORTANT: Check meta tags (Google Patents uses these!)
    meta_inventors = soup.select("meta[name='citation_author']")
    for meta in meta_inventors:
        content = meta.get("content")
        if content:
            inventors.add(content.strip())
    
    return "; ".join(list(inventors)) if inventors else None


# =================================================
# ASSIGNEE
# =================================================

def extract_assignees(soup):

    assignees = set()

    selectors = [

        "[itemprop='assigneeOriginal']",
        "[itemprop='assigneeCurrent']",
        "meta[scheme='assignee']",
        "dd[itemprop*='assignee']"
    ]

    for selector in selectors:

        elements = soup.select(selector)

        for e in elements:

            txt = e.get("content", "").strip()

            if not txt:
                txt = e.get_text(
                    " ",
                    strip=True
                )

            if txt:
                assignees.add(txt)

    return "; ".join(list(assignees))


# =================================================
# CITATIONS
# =================================================

def extract_backward_citation_count(driver):

    try:

        citations = driver.find_elements(
            By.CSS_SELECTOR,
            "state-modifier[data-result*='patent/']"
        )

        return len(citations)

    except:
        return 0


def extract_forward_citations(driver):

    try:

        forward = driver.find_elements(
            By.CSS_SELECTOR,
            "tr[itemprop='forwardReferencesFamily']"
        )

        return len(forward)

    except:
        return 0






def extract_legal_status(soup):
    # First check meta tag
    meta_status = soup.select_one("meta[name='citation_status']")
    if meta_status and meta_status.get("content"):
        return meta_status.get("content").strip()
    
    # Then try HTML elements
    status_selectors = [
        "span[itemprop='status']",
        ".status",
        "div.legal-status",
        "span.legal-status",
        "div.status",
        ".legal-status-text"
    ]
    
    for selector in status_selectors:
        try:
            el = soup.select_one(selector)
            if el:
                txt = el.get_text(strip=True)
                if txt and len(txt) > 1:
                    # Clean up common statuses
                    if "Active" in txt or "Inactive" in txt or "Expired" in txt:
                        return txt
                    return txt
        except:
            continue
    
    return None




def scrape_patent(driver, url):
    print("\nOpening:", url)
    driver.get(url)
    
    wait = WebDriverWait(driver, 60)
    
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        print("Body loaded")
    except TimeoutException:
        print("Timeout loading page:", url)
        return None
    
    # Give Google Patents extra time
    time.sleep(0.5)

    
    # Scroll to trigger lazy loading
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5)

    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.5)

    
    patent_id = extract_patent_id(url)
    
    # # Save debug HTML
    # with open(f"debug_{patent_id}.html", "w", encoding="utf-8") as f:
    #     f.write(driver.page_source)
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    data = {
        "source_url": url,
        "patent_id": patent_id,
        "title": None,
        "title_en": None,
        "abstract": None,
        "abstract_en": None,
        "description": None,
        "claims": None,
        "claims_en": None,
        "filing_date": None,
        "publication_date": None,
        "country": extract_country(patent_id),
        "language": None,
        "assignee_original": None,
        "assignee_en": None,
        "inventors": None,
        "ipc_codes": [],
        "cpc_codes": [],
        "claim_count": 0,
        "independent_claim_count": 0,
        "backward_citation_count": 0,
        "forward_citation_count": 0,
        "legal_status": None
    }
    
    # ============================================
    # FIX 1: TITLE EXTRACTION - Multiple selectors
    # ============================================
    title_selectors = [
        "h1#title",
        "h1[itemprop='title']",
        "h1.title",
        "meta[property='og:title']",
        "meta[name='DC.title']",
        "title"
    ]
    
    for selector in title_selectors:
        try:
            if selector.startswith("meta"):
                el = soup.select_one(selector)
                if el and el.get("content"):
                    data["title"] = el.get("content").strip()
                    break
            else:
                el = soup.select_one(selector)
                if el:
                    data["title"] = el.get_text(" ", strip=True)
                    if data["title"] and len(data["title"]) > 5:
                        break
        except:
            continue
    
    print(f"Title: {data['title'][:100] if data['title'] else 'None'}...")
    
    # ============================================
    # FIX 2: ABSTRACT EXTRACTION
    # ============================================
    abstract_selectors = [
        "section[itemprop='abstract']",
        "div.abstract",
        "meta[name='DC.description']",
        "meta[name='description']",
        "div#abstract",
        "abstract"
    ]
    
    for selector in abstract_selectors:
        try:
            if selector.startswith("meta"):
                el = soup.select_one(selector)
                if el and el.get("content"):
                    data["abstract"] = el.get("content").strip()
                    if data["abstract"] and len(data["abstract"]) > 10:
                        break
            else:
                el = soup.select_one(selector)
                if el:
                    data["abstract"] = el.get_text(" ", strip=True)
                    if data["abstract"] and len(data["abstract"]) > 10:
                        break
        except:
            continue
    
    print(f"Abstract: {data['abstract'][:100] if data['abstract'] else 'None'}...")
    
    # ============================================
    # FIX 3: INVENTORS EXTRACTION
    # ============================================
   # Inventors - Use the updated function
    data["inventors"] = extract_inventors(soup)
    print(f"Inventors: {data['inventors']}") 
    # ============================================
    # FIX 4: ASSIGNEE EXTRACTION
    # ============================================
    assignees = set()
    assignee_selectors = [
        "[itemprop='assigneeOriginal']",
        "[itemprop='assigneeCurrent']",
        ".assignee",
        ".current-assignee",
        "span.assignee"
    ]
    
    for selector in assignee_selectors:
        try:
            elements = soup.select(selector)
            for el in elements:
                txt = el.get_text(" ", strip=True)
                if txt and len(txt) > 2:
                    assignees.add(txt)
            if assignees:
                break
        except:
            continue
    
    # Also check meta tags
    meta_assignee = soup.select_one("meta[scheme='assignee']")
    if meta_assignee and meta_assignee.get("content"):
        assignees.add(meta_assignee.get("content").strip())
    
    data["assignee_original"] = "; ".join(list(assignees)) if assignees else None
    print(f"Assignee: {data['assignee_original']}")
    
    # ============================================
    # FIX 5: DATE EXTRACTION
    # ============================================
    # Filing Date
# Dates - Use the updated functions
    data["filing_date"] = extract_filing_date(soup)
    data["publication_date"] = extract_publication_date(soup)
    print(f"Filing: {data['filing_date']}, Publication: {data['publication_date']}")
    
    # ============================================
    # FIX 6: IPC/CPC EXTRACTION
    # ============================================
    ipc_set = set()
    cpc_set = set()
    
    # Try different selectors
    classification_selectors = [
        "classification-tree span",
        ".classification",
        "span.classification",
        "div.cpc",
        "div.ipc"
    ]
    
    for selector in classification_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements:
                txt = el.text.strip()
                # Look for IPC codes (A-H, Y)
                matches = re.findall(r'[A-HY]\d{2}[A-Z]\d+/\d+', txt)
                for m in matches:
                    ipc_set.add(m)
                    cpc_set.add(m)
            if ipc_set:
                break
        except:
            pass
    
    # Also try BeautifulSoup for classifications
    if not ipc_set:
        for el in soup.select(".classification, .cpc, .ipc"):
            txt = el.get_text(" ", strip=True)
            matches = re.findall(r'[A-HY]\d{2}[A-Z]\d+/\d+', txt)
            for m in matches:
                ipc_set.add(m)
                cpc_set.add(m)
    
    data["ipc_codes"] = list(ipc_set)
    data["cpc_codes"] = list(cpc_set)
    print(f"IPC codes: {len(data['ipc_codes'])}")
    
    # ============================================
    # FIX 7: LEGAL STATUS
    # ============================================
    # Legal status - Use the updated function
    data["legal_status"] = extract_legal_status(soup)
    print(f"Legal status: {data['legal_status']}")
    
    # ============================================
    # FIX 8: DESCRIPTION
    # ============================================
    # Description - Use the updated function
    data["description"] = extract_description(soup)
    
    # ============================================
    # FIX 9: CLAIMS (existing code with fallback)
    # ============================================
    claims, count, indep = extract_claims(soup)
    
    if count == 0:
        print(f"No claims found for {patent_id}. Trying fallback...")
        claims, count, indep = extract_claims_from_google(patent_id)
    
    data["claims"] = claims
    data["claim_count"] = count
    data["independent_claim_count"] = indep
    data["title_en"] = data["title"]
    data["abstract_en"] = data["abstract"]
    data["claims_en"] = data["claims"]
    data["assignee_en"] = data["assignee_original"]


    
    # ============================================
    # FIX 12: CITATIONS
    # ============================================
    try:
        citations = driver.find_elements(By.CSS_SELECTOR, "state-modifier[data-result*='patent/']")
        data["backward_citation_count"] = len(citations)
    except:
        data["backward_citation_count"] = 0
    
    try:
        forward = driver.find_elements(By.CSS_SELECTOR, "tr[itemprop='forwardReferencesFamily']")
        data["forward_citation_count"] = len(forward)
    except:
        data["forward_citation_count"] = 0
    
    print(f"Patent: {data['patent_id']}")
    print(f"Claims: {data['claim_count']}")
    print(f"Title: {data['title'] is not None}")
    print(f"Abstract: {data['abstract'] is not None}")
    print(f"Inventors: {data['inventors'] is not None}")
    print(f"Filing date: {data['filing_date'] is not None}")
    print("-" * 50)
    
    return data
# =================================================
# SCRAPE ALL
# =================================================

def scrape_all(url_list):

    driver = get_driver()

    results = []

    for url in tqdm(url_list):

        try:

            patent = scrape_patent(
                driver,
                url
            )

            results.append(patent)

            time.sleep(0.3)

        except Exception as e:

            print("ERROR:", url)
            print(e)

    driver.quit()

    return results


# =================================================
# MAIN
# =================================================

if __name__ == "__main__":

    # df_urls = pd.read_excel(
    #     r"D:\Patent-Intelligence\patents_url_0-502.xlsx"
    # )

    # url_list = (
    #     df_urls["url"]
    #     .dropna()
    #     .tolist()
    # )

    url_list = [
        "https://patents.google.com/patent/US12059124B2/en",
        "https://patents.google.com/patent/US20240000521A1/en",
        "https://patents.google.com/patent/US20220331018A1/en",
        "https://patents.google.com/patent/US12059218B2/en",
        "https://patents.google.com/patent/EP3506317B1/en"
 ]

    print(
        f"Loaded {len(url_list)} URLs"
    )

    results = scrape_all(url_list)

    with open(
        "patents.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    df = pd.DataFrame(results)

    # df.to_csv(
    #     "patents.csv",
    #     index=False,
    #     encoding="utf-8",
    #     quoting=csv.QUOTE_ALL
    # )

    # print(df.head())

# =================================================
# ROBUST SELECTOR HELPERS (ADD THESE TO IMPROVE ACCURACY)
# =================================================

def get_first_text(soup, selectors):
    for selector in selectors:
        try:
            el = soup.select_one(selector)
            if el:
                if el.name == "meta":
                    txt = el.get("content", "").strip()
                else:
                    txt = el.get_text(" ", strip=True)
                if txt:
                    return re.sub(r'\s+', ' ', txt)
        except:
            pass
    return None

def extract_title_robust(soup):
    return get_first_text(soup, [
        "h1#title",
        "h1[itemprop='title']",
        "meta[name='DC.title']",
        "meta[property='og:title']",
        "title"
    ])

def extract_abstract_robust(soup):
    return get_first_text(soup, [
        "meta[name='DC.description']",
        "meta[name='description']",
        "section[itemprop='abstract']",
        "div.abstract",
        "abstract"
    ])

