"""
huggingface_blog_scraper_stable.py

Minimal changes from your original code:
- keep get_driver(), get_links_from_every_page(), scrape_articles(urls)
- save links to a separate file for later use
- add a global retry loop until all articles are scraped or max rounds reached
- keep multiprocessing + checkpointing + error persistence
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Dict, Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from newspaper import Article, Config
import requests

# ------------- Global session (not strictly required but fine) -------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 ... Safari/537.36"
})

# ------------- Newspaper config defaults -------------
base_config = Config()
base_config.request_timeout = 10
base_config.fetch_images = False
base_config.memoize_articles = False

# ------------------ Configuration ------------------
CHECKPOINT_FILE = "hg_blogs_checkpoint.json"
OUTPUT_FILE = "hg_blogs_data.json"
LINKS_FILE = "hf_blogs_links.json"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("hf_scraper")

# ------------------ get_driver (same essence) ------------------
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

    driver.implicitly_wait(10)
    return driver


def get_links_first_page() -> List[str]:
    """
    Fetch only the first huggingface blog page (p=0) and return the links found.
    Uses the same 'shadow-xs' selector you rely on so behaviour is consistent.
    """
    driver = None
    links = []
    try:
        driver = get_driver()
        url = "https://huggingface.co/blog?p=0"
        logger.info(f"Fetching first page for quick check: {url}")
        try:
            driver.get(url)
        except WebDriverException as e:
            logger.error(f"Driver failed to load first page {url}: {e}")
            return []

        wait = WebDriverWait(driver, 10)
        try:
            elems = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "shadow-xs")))
        except TimeoutException:
            logger.warning("No 'shadow-xs' elements found on first page.")
            return []
        except Exception as e:
            logger.warning(f"Error locating link elements on first page: {e}")
            return []

        for el in elems:
            try:
                href = el.get_attribute("href")
            except Exception:
                href = None
            if href:
                links.append(href)

        # dedupe preserve order
        links = list(dict.fromkeys(links))
        logger.info(f"Found {len(links)} links on first page.")
        return links
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

# ------------------ Links collector (same logic, one driver) ------------------
def get_links_from_every_page(start_page: int = 0) -> List[str]:
    """
    Collect all article links paginated at https://huggingface.co/blog?p={page}
    Uses the same class 'shadow-xs' that you used originally.
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
                links = wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "shadow-xs"))
                )
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

            # respect the site a bit
            time.sleep(random.uniform(0.5, 1.5))
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    unique_links = list(set(all_links))
    logger.info(f"Total unique article links collected: {len(unique_links)}")
    return unique_links


def check_and_update_links_top_page(links_file: str = LINKS_FILE) -> Tuple[List[str], List[str]]:
    """
    Assumes links_file exists (first-run check is done by caller).
    1) Load existing links.
    2) Scrape first page links (top_links).
    3) Compare top_links with existing[:len(top_links)].
       If union size equals length of existing[:len(top_links)] then no new links.
       Otherwise prepend only the new top-links to the saved file.
    Returns: (updated_links_in_memory, new_links_found)
    """
    # load existing links (caller ensured file exists, but we still handle errors)
    existing = load_links(links_file)

    # fetch only first page links
    top_links = get_links_first_page()
    if not top_links:
        logger.info("No links found on first page; leaving links file unchanged.")
        return existing, []

    # compare only against top N of existing where N = number of links found on first page
    n = len(top_links)
    existing_top_n = existing[:n]

    # union of sets (robust and simple)
    union_len = len(set(top_links) | set(existing_top_n))

    # if union length equals length of existing_top_n -> no new links (no change)
    if union_len == len(existing_top_n):
        logger.info("No new links on first page compared to saved top-N. No update needed.")
        return existing, []

    # otherwise find which top page links are not already present in the saved top-N
    new_links = [l for l in top_links if l not in existing_top_n]
    if not new_links:
        # defensive: if union_len differs for some weird reason but no new links, treat as no-change
        logger.info("Union indicated change but no new links found (odd). No update.")
        return existing, []

    # prepend the new links (preserve order: newest first as they appear on first page)
    updated = list(dict.fromkeys(new_links + existing))  # dedupe and keep order (new first)
    try:
        with open(links_file, "w") as f:
            json.dump(updated, f, indent=2)
        logger.info(f"Prepended {len(new_links)} new links to {links_file}. Total now: {len(updated)}")
    except Exception as e:
        logger.error(f"Failed to write updated links file {links_file}: {e}")
        # still return the computed in-memory values
    return updated, new_links


# ------------------ Save/load links ------------------
def save_links(links: List[str], path: str = LINKS_FILE):
    try:
        unique_links = list(dict.fromkeys(links))
        with open(path, "w") as f:
            json.dump(unique_links, f, indent=2)
        logger.info(f"Saved {len(unique_links)} links to {path}")
    except Exception as e:
        logger.error(f"Failed to save links to {path}: {e}")

def load_links(path: str = LINKS_FILE) -> List[str]:
    try:
        with open(path, "r") as f:
            links = json.load(f)
        logger.info(f"Loaded {len(links)} links from {path}")
        return links
    except FileNotFoundError:
        logger.warning(f"Links file {path} not found.")
        return []
    except Exception as e:
        logger.error(f"Failed to load links from {path}: {e}")
        return []

# ------------------ Worker (multiprocessing) ------------------
def _article_worker(url: str, do_nlp: bool = True, retries: int = 2) -> Dict:
    """
    Single-URL worker. Designed to be run in a separate process.
    Returns a dictionary similar to your original structure.
    """
    last_exc = None
    for attempt in range(1, retries + 2):
        # rotate UA per attempt
        UAS = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        ]
        config = Config()
        config.request_timeout = base_config.request_timeout
        config.fetch_images = base_config.fetch_images
        config.memoize_articles = base_config.memoize_articles
        config.browser_user_agent = random.choice(UAS)

        try:
            # if attempt == retries + 1:
            #     time.sleep(10)  # extra wait on final attempt

            # IMPORTANT: actually use the config now
            article = Article(url, config=config)
            article.download()
            article.parse()

            data = {
                "Title": article.title or "",
                "Authors": article.authors or [],
                "Publish Date": article.publish_date.isoformat() if article.publish_date else None,
                "Text": article.text or "",
                "link": url,
                "scraped_at": datetime.utcnow().isoformat() + "Z",
            }

            if do_nlp:
                try:
                    article.nlp()
                    data["Summary"] = article.summary or ""
                    data["Keywords"] = article.keywords or []
                except Exception as nlp_e:
                    logger.debug(f"NLP failed for {url}: {nlp_e}")
                    data["Summary"] = ""
                    data["Keywords"] = []
                    data["_nlp_error"] = str(nlp_e)

            # polite jitter
            time.sleep(random.uniform(0.5, 1.5))
            return data

        except Exception as e:
            last_exc = e
            logger.warning(f"Attempt {attempt} failed to download/parse {url}: {e}")
            time.sleep(3 * attempt + random.random())

    # If it reaches here, all retries failed - raise so caller can log/store failure
    raise last_exc

# ------------------ scrape_articles (same API, one round) ------------------
def scrape_articles(
    urls: Iterable[str],
    out_checkpoint: str = CHECKPOINT_FILE,
    max_workers: int = 6,
    resume: bool = True,
    do_nlp: bool = True
) -> List[Dict]:
    """
    One scraping round over given URLs.

    - Keeps your original function name and top-level usage.
    - Uses multiprocessing + checkpoint.
    - Returns a list where some entries may still have '_error'.
    """
    urls = list(dict.fromkeys(urls))
    logger.info(f"Starting scrape for {len(urls)} URLs (workers={max_workers})")

    results: List[Dict] = []
    processed_links = set()

    # load checkpoint if requested (for this round)
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
    logger.info(f"{len(to_process)} URLs remain to process after resume/dedupe for this round.")

    if not to_process:
        logger.info("Nothing to process for this round; returning checkpoint contents.")
        return results

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

            # write checkpoint incrementally for that round
            try:
                with open(out_checkpoint, "w") as ck:
                    json.dump(results, ck, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write checkpoint file {out_checkpoint}: {e}")

    logger.info("Scraping round finished.")
    return results

# ------------------ Global retry orchestrator ------------------
def scrape_all_with_retries(
    urls: Iterable[str],
    max_rounds: int = 5,
    max_workers: int = 6,
    do_nlp: bool = True
) -> (List[Dict], List[str]):
    """
    Orchestrates multiple scraping rounds until:
      - all URLs have been scraped with no '_error', or
      - max_rounds is reached.

    Returns:
      final_results: list[dict] (last result per link)
      remaining_failed_links: list[str] (links still with _error after all rounds)
    """
    all_urls = list(dict.fromkeys(urls))
    results_by_link: Dict[str, Dict] = {u: {} for u in all_urls}
    remaining = all_urls[:]

    for round_idx in range(1, max_rounds + 1):
        if not remaining:
            logger.info(f"No remaining URLs to scrape. Stopping after {round_idx - 1} rounds.")
            break

        logger.info(f"=== Global scraping round {round_idx} with {len(remaining)} URLs ===")
        # For each global round, we can ignore previous checkpoint resume (we want fresh tries)
        round_results = scrape_articles(
            remaining,
            out_checkpoint=CHECKPOINT_FILE,
            max_workers=max_workers,
            resume=False,
            do_nlp=do_nlp,
        )

        # Update latest result per link
        for item in round_results:
            link = item.get("link")
            if not link:
                continue
            results_by_link[link] = item

        # Determine which still failed
        remaining = [
            link for link, item in results_by_link.items()
            if not item or "_error" in item
        ]

        if remaining:
            logger.warning(f"{len(remaining)} URLs still failing after round {round_idx}. Retrying...")
            # a little pause before next global round
            time.sleep(5 * round_idx)
        else:
            logger.info("All URLs scraped successfully; no remaining errors.")
            break

    final_results = list(results_by_link.values())
    failed_links = [link for link, item in results_by_link.items() if "_error" in item]
    return final_results, failed_links

# ------------------ Save helper ------------------
def save_json(data: List[Dict], path: str = OUTPUT_FILE):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} articles to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")

# ------------------ Main script flow ------------------
if __name__ == "__main__":
    # 1) collect links (once) and save them
    if os.path.exists(LINKS_FILE):
        check_and_update_links_top_page()

    all_links = get_links_from_every_page()
    save_links(all_links, LINKS_FILE)

    # 2) scrape all with repeated rounds until everything is clean or max_rounds hit
    articles, failed_links = scrape_all_with_retries(
        all_links,
        max_rounds=5,   # you can bump this to 999 if you really want "until success"
        max_workers=4,  # tune as per your CPU / network
        do_nlp=True,
    )

    # 3) save final JSON
    save_json(articles, OUTPUT_FILE)

    # 4) report status
    if failed_links:
        logger.warning(
            f"{len(failed_links)} articles still failed after all rounds. "
            f"Check {OUTPUT_FILE} for _error fields."
        )
    else:
        logger.info("All articles scraped successfully. No _error entries in JSON.")
