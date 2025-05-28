import requests
import re
from bs4 import BeautifulSoup


def get_documentation_urls():
    return [
        '/docs/huggingface_hub/guides/overview',
        '/docs/huggingface_hub/guides/download',
        '/docs/huggingface_hub/guides/upload',
        '/docs/huggingface_hub/guides/hf_file_system',
        '/docs/huggingface_hub/guides/repository',
        '/docs/huggingface_hub/guides/search',
        # You may add additional URLs here or replace all of them
    ]


def construct_full_url(base_url, relative_url):
    return base_url + relative_url


def scrape_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.body.text.strip()
    # Remove non-ASCII characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def scrape_all_content(base_url, relative_urls, filename):
    content = []
    for relative_url in relative_urls:
        full_url = construct_full_url(base_url, relative_url)
        scraped_content = scrape_page_content(full_url)
        content.append(scraped_content.rstrip('\n'))

    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n" % item)

    return content

scrape_all_content(
    base_url='https://huggingface.co',
    relative_urls=get_documentation_urls(),
    filename='huggingface_docs.txt'
)
