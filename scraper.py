"""
huggingface_blog_scraper_stable.py

Minimal changes from user's original code:
- keep get_driver(), get_links_from_every_page(), scrape_articles(urls)
- add logging, retries, checkpointing, multiprocessing, better exception handling
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Dict, Iterable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# ------------------ Configuration ------------------
CHECKPOINT_FILE = "hg_blogs_checkpoint.json"
OUTPUT_FILE = "hg_blogs_data.json"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("hf_scraper")

# ------------------ get_driver (kept similar to yours) ------------------
def get_driver() -> webdriver.Chrome:
    chrome_options = Options()

    # ------------ HEADLESS MODE ------------
    chrome_options.add_argument("--headless=new")   # modern headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    # ------------ PERFORMANCE / STABILITY ------------
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")

    # ------------ STEALTH (Avoid detection) ------------
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    # ------------ LANGUAGE / LOCALE ------------
    chrome_options.add_argument("--lang=en-US")
    chrome_options.add_argument(f"user-agent={USER_AGENT}")

    # ------------ CREATE DRIVER ------------
    # If chromedriver isn't on PATH, pass Service(executable_path="/path/to/chromedriver")
    driver = webdriver.Chrome(service=Service(), options=chrome_options)

    # ------------ EXTRA STEALTH JS PATCH ------------
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            }
        )
    except Exception:
        logger.debug("Stealth JS injection not applied (non-fatal).")

    # ------------ GLOBAL WAIT SETUP ------------
    driver.implicitly_wait(10)

    return driver

# ------------------ Links collector (kept your behavior) ------------------
def get_links_from_every_page(start_page: int = 0) -> List[str]:
    """
    Collect all article links paginated at https://huggingface.co/blog?p={page}
    Preserves the core behavior and element class you used originally ("shadow-xs").
    Uses a single driver for the whole run (keeps your function name/semantics but speeds up).
    """
    all_links = []
    page_number = start_page
    driver = None

    try:
        driver = get_driver()
        while True:
            hf_blogs_link = f"https://huggingface.co/blog?p={page_number}"
            logger.info(f"Getting links from page: {page_number} -> {hf_blogs_link}")
            try:
                driver.get(hf_blogs_link)
            except WebDriverException as e:
                logger.error(f"Driver failed to load page {hf_blogs_link}: {e}")
                break

            wait = WebDriverWait(driver, 10)
            try:
                # Keep same selector/class you used originally.
                links = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "shadow-xs")))
            except TimeoutException:
                logger.info("End of pages reached (no 'shadow-xs' elements found).")
                break
            except Exception as e:
                logger.warning(f"Unexpected error while locating elements on page {page_number}: {e}")
                break

            added = 0
            for link in links:
                try:
                    href = link.get_attribute("href")
                except Exception as e:
                    logger.debug(f"Could not get href from element: {e}")
                    href = None
                if href:
                    all_links.append(href)
                    added += 1

            logger.info(f"Found {added} links on page {page_number}.")
            page_number += 1

            # Respectful pause to avoid hammering the server
            time.sleep(random.uniform(0.8, 1.6))
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    # dedupe while preserving order
    unique_links = list(dict.fromkeys(all_links))
    logger.info(f"Total unique article links collected: {len(unique_links)}")
    return unique_links

# ------------------ Worker used by multiprocessing (keeps newspaper usage) ------------------
def _article_worker(url: str, do_nlp: bool = True, retries: int = 2) -> Dict:
    """
    Single-URL worker. Designed to be run in a separate process.
    Returns a dictionary similar to your original structure.
    """
    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            article = Article(url)
            article.download()
            article.parse()

            data = {
                "Title": article.title or "",
                "Authors": article.authors or [],
                "Publish Date": article.publish_date.isoformat() if article.publish_date else None,
                "Text": article.text or "",
                "link": url
                   }

            if do_nlp:
                try:
                    article.nlp()
                    # avoid accidental tuple (keep exactly as your original code intended)
                    data["Summary"] = article.summary or ""
                    data["Keywords"] = article.keywords or []
                except Exception as nlp_e:
                    logger.debug(f"NLP failed for {url}: {nlp_e}")
                    data["Summary"] = ""
                    data["Keywords"] = []
                    data["_nlp_error"] = str(nlp_e)

            # polite jitter
            time.sleep(random.uniform(2, 4))
            return data

        except Exception as e:
            last_exc = e
            logger.warning(f"Attempt {attempt} failed to download/parse {url}: {e}")
            # backoff
            time.sleep(1.2 * attempt + random.random())

    # If it reaches here, all retries failed - raise so caller can log/store failure
    raise last_exc

# ------------------ scrape_articles (keeps name & overall shape) ------------------
def scrape_articles(urls: Iterable[str],
                    out_checkpoint: str = CHECKPOINT_FILE,
                    max_workers: int = 6,
                    resume: bool = True,
                    do_nlp: bool = True) -> List[Dict]:
    """
    Top-level function that mirrors your original call signature:
        articles = scrape_articles(all_links)

    Enhancements:
    - Concurrent processing using ProcessPoolExecutor.
    - Checkpointing to resume if interrupted.
    - On permanent failure we store an entry with _error so you still have link covered.
    """
    urls = list(dict.fromkeys(urls))  # dedupe while preserving order
    logger.info(f"Starting scrape for {len(urls)} URLs (workers={max_workers})")

    results = []
    processed_links = set()

    # load checkpoint if requested
    if resume and os.path.exists(out_checkpoint):
        try:
            with open(out_checkpoint, "r") as f:
                checkpoint_data = json.load(f)
            for item in checkpoint_data:
                results.append(item)
                if isinstance(item, dict) and "link" in item:
                    processed_links.add(item["link"])
            logger.info(f"Loaded {len(results)} items from checkpoint {out_checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {out_checkpoint}: {e}")

    to_process = [u for u in urls if u not in processed_links]
    logger.info(f"{len(to_process)} URLs remain to process after resume/dedupe.")

    if not to_process:
        logger.info("Nothing to process; returning checkpoint contents.")
        return results

    # Use processes so newspaper parsing runs without GIL limitations and isolates issues
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_article_worker, url, do_nlp): url for url in to_process}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                data = fut.result()
                results.append(data)
                logger.info(f"Scraped: {url}")
            except Exception as e:
                logger.error(f"Permanent failure for {url}: {e}")
                results.append({"link": url, "_error": str(e)})
            # write checkpoint incrementally (safe-ish)
            try:
                with open(out_checkpoint, "w") as ck:
                    json.dump(results, ck, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write checkpoint file {out_checkpoint}: {e}")

    logger.info("All scraping tasks finished.")
    return results

# ------------------ Save helper (keeps yours) ------------------
def save_json(data: List[Dict], path: str = OUTPUT_FILE):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} articles to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")

# ------------------ Run example (keeps similar flow) ------------------
if __name__ == "__main__":
    # 1) collect links (this function name / semantics preserved)
    all_links = get_links_from_every_page()

    # 2) scrape articles (same top-level function name you started with)
    articles = scrape_articles(all_links, max_workers=3, resume=True)

    # 3) save to JSON
    save_json(articles, OUTPUT_FILE)
