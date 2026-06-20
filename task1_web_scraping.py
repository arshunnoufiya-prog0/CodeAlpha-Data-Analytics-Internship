"""
=============================================================
CodeAlpha Internship — Task 1: Web Scraping
=============================================================
Scrapes book data from books.toscrape.com (a legal practice site)
Collects: Title, Price, Rating, Availability
Saves results to CSV for use in Tasks 2 & 3
=============================================================
Install dependencies:
    pip install requests beautifulsoup4 pandas
=============================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://books.toscrape.com/catalogue/"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url):
    """Fetch a page and return a BeautifulSoup object."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_books(soup):
    """Extract book data from a single catalogue page."""
    books = []
    for article in soup.select("article.product_pod"):
        title = article.h3.a["title"]
        price = article.select_one("p.price_color").text.strip().replace("Â", "")
        rating_word = article.p["class"][1]          # e.g. "Three"
        rating = RATING_MAP.get(rating_word, 0)
        availability = article.select_one("p.availability").text.strip()
        books.append({
            "Title": title,
            "Price (£)": float(price.replace("£", "")),
            "Rating (1-5)": rating,
            "Availability": availability,
        })
    return books


def get_next_page(soup):
    """Return the URL of the next page, or None if last page."""
    next_btn = soup.select_one("li.next > a")
    if next_btn:
        return BASE_URL + next_btn["href"]
    return None


def scrape_all_books(max_pages=10):
    """Scrape up to max_pages pages and return all books."""
    all_books = []
    url = START_URL
    page = 1

    while url and page <= max_pages:
        print(f"  Scraping page {page}: {url}")
        soup = get_soup(url)
        all_books.extend(parse_books(soup))
        url = get_next_page(soup)
        page += 1
        time.sleep(0.5)   # be polite — don't hammer the server

    return all_books


def main():
    print("=" * 55)
    print("  CodeAlpha Task 1 — Web Scraping")
    print("  Source : books.toscrape.com")
    print("=" * 55)

    books = scrape_all_books(max_pages=10)   # ~200 books (20/page)

    df = pd.DataFrame(books)
    df.index += 1                            # start index from 1

    # ── Basic summary ──────────────────────────────────────
    print(f"\n✅ Total books scraped : {len(df)}")
    print(f"   Price range        : £{df['Price (£)'].min():.2f} – £{df['Price (£)'].max():.2f}")
    print(f"   Avg rating         : {df['Rating (1-5)'].mean():.2f} / 5")
    print(f"\n{df.head(10).to_string()}\n")

    # ── Save to CSV ────────────────────────────────────────
    output_file = "books_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Dataset saved to '{output_file}'")
    print("   Use this CSV in Task 2 (EDA) and Task 3 (Visualization).")


if __name__ == "__main__":
    main()
