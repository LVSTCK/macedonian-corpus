import os
import requests
from bs4 import BeautifulSoup


def download_pdf(pdf_url, save_path, chunk_size=None):
    """
    Download a PDF from the given URL and save it to the specified path.

    Parameters:
    -----------
    pdf_url : str
        The URL of the PDF file to download.
    save_path : str
        The local path where the PDF file will be saved.
    chunk_size : int, optional
        The size of the chunks to use for downloading. If None, the file
        will be downloaded without chunking.
    """
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # Raise an error if the request failed

        with open(save_path, "wb") as pdf_file:
            if chunk_size:
                # Download in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        pdf_file.write(chunk)
            else:
                # Download the entire content at once
                pdf_file.write(response.content)

        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")


def fetch_pdfs_recursive(
    base_url,
    output_dir,
    current_url=None,
    visited_urls=None,
    pdf_chunk_size=None,
):
    """
    Recursively crawls a website to extract and download all PDF links.

    Starting from the `base_url`, the function recursively visits all internal
    links within the same domain, identifies PDF links, and downloads them to
    the specified `output_dir`. It tracks visited URLs to avoid revisiting and
    skips non-HTML content like audio files.

    Parameters:
    -----------
    base_url : str
        The starting URL to restrict crawling to the same domain.
    output_dir : str
        Directory to save downloaded PDFs; created if it does not exist.
    current_url : str, optional
        The current URL being processed (default: `base_url`).
    visited_urls : set, optional
        Set of visited URLs to prevent revisiting (default: empty set).
    chunk_size : int, optional
        The size of the chunks to use for downloading PDFs. If None, the file
        will be downloaded without chunking.
    """
    if current_url is None:
        current_url = base_url

    if visited_urls is None:
        visited_urls = set()

    os.makedirs(output_dir, exist_ok=True)

    print(f"Visiting: {current_url}")
    visited_urls.add(current_url)

    try:
        # Send a GET request to the current URL
        response = requests.get(current_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all <a> tags with href attributes
        links = soup.find_all("a", href=True)
        filtered_links = [
            link["href"] for link in links if "http" in link["href"]
        ]
        for link in filtered_links:
            if link.endswith(".pdf"):
                # If it's a PDF link download it
                filename = link.split("/")[-1]

                print(f"Downloading: {filename}\n")
                download_pdf(
                    link, os.path.join(output_dir, filename), pdf_chunk_size
                )
            elif (
                base_url in link
                and link not in visited_urls
                and "ogg" not in link
            ):
                # Skip ogg (audio files) since they slow down the process
                fetch_pdfs_recursive(
                    base_url=base_url,
                    output_dir=output_dir,
                    current_url=link,
                    visited_urls=visited_urls,
                    pdf_chunk_size=pdf_chunk_size,
                )

    except Exception as e:
        print(f"Error while processing {current_url}: {e}")


if __name__ == "__main__":
    base_url = "https://indibib.feit.ukim.edu.mk"
    output_dir = "../data/ukim_library"
    fetch_pdfs_recursive(base_url, output_dir)
