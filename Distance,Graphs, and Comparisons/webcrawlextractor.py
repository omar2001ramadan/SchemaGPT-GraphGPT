import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_links(url):
    urls = set()
    domain_name = urlparse(url).netloc
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        if not is_valid_url(href):
            continue
        if domain_name not in parsed_href.netloc:
            continue
        urls.add(href)
    return urls

def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    return text

def crawl_website(start_url):
    visited_urls = set()
    urls_to_visit = [start_url]
    all_text = ""

    while urls_to_visit:
        url = urls_to_visit.pop(0)
        if url in visited_urls:
            continue
        visited_urls.add(url)
        print(f"Crawling: {url}")
        text = extract_text(url)
        all_text += text + "\n\n"
        new_urls = get_all_links(url)
        urls_to_visit.extend(new_urls - visited_urls)

    return all_text

def save_text_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)

if __name__ == "__main__":
    start_url = "https://yworks.github.io/yfiles-jupyter-graphs/"  # Replace with the target website URL-----------------------------------------<<<<<<<<<<
    output_file = "y file documentation" # Name of output ---------------------------------------------------------------------------------------<<<<<<<<<<
    website_text = crawl_website(start_url)
    save_text_to_file(website_text, output_file)
    print(f"All text extracted and saved to {output_file}")
