import asyncio
import aiohttp
import re
import aiofiles
import os
import logging
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse, parse_qs, urljoin

# -------------------- Configure Logging --------------------
logger = logging.getLogger(__name__)
logger.setLevel(
    logging.DEBUG
)  # Capture all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Console handler (shows INFO+ messages on console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

IGNORED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
    "ogg",
    ".zip",
    ".rar",
    ".tar",
    ".gz",
    ".csv",
    ".xlsx",
)


async def download_pdf_async(pdf_url, session, save_dir, chunk_size=None):
    """
    Download a PDF from .php or .pdf links, ensuring a unique filename if
    the server does not provide one.
    """
    try:
        async with session.get(pdf_url) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            content_disp = response.headers.get("Content-Disposition", "")
            fname_match = re.search(r'filename="([^"]+)"', content_disp)

            if fname_match:
                # If server sent a filename
                filename = fname_match.group(1)
            else:
                # Parse from URL
                parsed_url = urlparse(pdf_url)
                fallback_name = os.path.basename(
                    parsed_url.path
                )  # e.g. 'download.php'
                fallback_name = unquote(fallback_name)

                # If there's a query param like id=16, use that
                query_params = parse_qs(parsed_url.query)
                file_id = query_params.get("id", ["unknown"])[0]

                # e.g. 'download_id_16.pdf'
                filename = f"{fallback_name}_id_{file_id}.pdf"

            # Always decode any leftover encodings
            filename = unquote(filename)

            # If server says PDF but no .pdf extension, add it
            if (
                "application/pdf" in content_type
                and not filename.lower().endswith(".pdf")
            ):
                filename += ".pdf"

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            # If we already have this file, skip
            if os.path.exists(save_path):
                logger.info(f"Already downloaded: {save_path}")
                return

            # Download logic
            if chunk_size:
                async with aiofiles.open(save_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(
                        chunk_size
                    ):
                        await f.write(chunk)
            else:
                data = await response.read()
                async with aiofiles.open(save_path, "wb") as f:
                    await f.write(data)

            logger.info(f"Downloaded: {save_path}")

    except Exception as e:
        logger.error(f"Failed to download {pdf_url}: {e}")


async def fetch_and_parse(session, url):
    """
    Fetch the resource at `url` asynchronously.
    If it's HTML (Content-Type: text/html), parse and return a
    BeautifulSoup object.
    Otherwise, return None.
    """
    async with session.get(url) as response:
        response.raise_for_status()

        # Check the Content-Type to see if it's HTML
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            text = await response.text()
            return BeautifulSoup(text, "html.parser")
        else:
            # Not HTML (could be PDF, ZIP, etc.)
            return None


async def fetch_pdfs(
    base_url,
    url,
    visited_urls,
    session,
    output_dir,
    max_depth=None,
    depth=0,
    chunk_size=None,
):
    """
    Recursively crawl `url`:
      - Download any .pdf or .php-based PDF (detected by "download.php") found.
      - Recursively visit child links in the same domain.
    """
    if max_depth is not None and depth > max_depth:
        return

    if url in visited_urls:
        return
    visited_urls.add(url)

    logger.info(f"Visiting: {url}")

    try:
        soup = await fetch_and_parse(session, url)
        # If soup is None, it wasn't HTML (likely a file), so skip link parsing
        if not soup:
            return

        # Extract <a> tags. Convert relative links to absolute via urljoin
        links = []
        for a_tag in soup.find_all("a", href=True):
            absolute_link = urljoin(url, a_tag["href"])
            links.append(absolute_link)

        tasks = []
        for link in links:
            link_lower = link.lower()

            # Skip links containing irrelevant file extensions
            if any(ext in link_lower for ext in IGNORED_EXTENSIONS):
                continue

            # If it's a .pdf or something like 'download.php'
            if ".pdf" in link_lower or "download.php" in link_lower:
                if link not in visited_urls:
                    visited_urls.add(link)
                    logger.info(f"Downloading PDF link: {link}")
                    tasks.append(
                        download_pdf_async(
                            pdf_url=link,
                            session=session,
                            save_dir=output_dir,
                            chunk_size=chunk_size,
                        )
                    )
            # Otherwise, if it’s in the same domain, recurse deeper
            elif base_url in link:
                tasks.append(
                    fetch_pdfs(
                        base_url,
                        link,
                        visited_urls,
                        session,
                        output_dir,
                        max_depth=max_depth,
                        depth=depth + 1,
                        chunk_size=chunk_size,
                    )
                )

        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Error processing {url}: {e}")


async def main():
    base_url = ""
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    visited_urls = set()

    # --- Setup File Logger ---
    log_file_path = os.path.join(output_dir, "scrape.log")
    file_handler = logging.FileHandler(
        log_file_path, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # log all levels to file
    file_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    # ---------------------------------------------------------

    CHUNK_SIZE = 4096  # or None

    async with aiohttp.ClientSession() as session:
        await fetch_pdfs(
            base_url=base_url,
            url=base_url,
            visited_urls=visited_urls,
            session=session,
            output_dir=output_dir,
            max_depth=None,  # If you want to limit recursion depth
            chunk_size=CHUNK_SIZE,
        )

    logger.info("Crawling complete.")


if __name__ == "__main__":
    asyncio.run(main())
