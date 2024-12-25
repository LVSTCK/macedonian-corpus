import asyncio
import aiohttp
import re
import aiofiles
import os
import logging
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse, parse_qs, urljoin
from playwright.async_api import async_playwright


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
    Download a PDF (or PDF served via .php) asynchronously.

    This function makes an HTTP GET request (via an existing aiohttp session)
    to retrieve the binary content of a PDF file, then writes it to disk.

    Parameters
    ----------
    pdf_url : str
        The direct link (URL) to the PDF or a server script returning a PDF.
    session : aiohttp.ClientSession
        An existing aiohttp.ClientSession to use for requests.
    save_dir : str
        The directory where the downloaded PDF file will be saved.
    chunk_size : int or None, optional
        If not None, download the PDF in chunks of this size (in bytes).
        If None, download the entire file at once.

    Returns
    -------
    None
        The file is written to `save_dir`; no value is returned.

    Notes
    -----
    - It automatically derives a filename, either from the server's
      Content-Disposition header or, if missing, from the URL path.
    - If the Content-Type is "application/pdf" but the filename does not end
      with ".pdf", the function appends ".pdf".
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

            # Decode leftover encodings
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


async def fetch_and_parse(session, url, render_js=False):
    """
    Fetch and parse an HTML page, optionally rendering JavaScript with
    Playwright.
    If ``render_js=False``, this function uses ``aiohttp`` to fetch the page.
    If ``render_js=True``, it launches a headless Chromium browser via
    Playwright, navigates to the page, waits until the network is idle,
    and then extracts the final rendered HTML.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The aiohttp session used if ``render_js=False``. For ``render_js=True``,
        the page is fetched by Playwright, but the session is kept.
    url : str
        The URL of the page to fetch.
    render_js : bool, optional
        Whether to fully render the page's JavaScript via Playwright.
        Defaults to False.

    Returns
    -------
    BeautifulSoup or None
        A BeautifulSoup object containing the page's HTML, or None if the
        content was not HTML or an error occurred.
    """
    if not render_js:
        # Original AIOHTTP approach
        async with session.get(url) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                text = await response.text()
                return BeautifulSoup(text, "html.parser")
            else:
                return None
    else:
        # Playwright approach
        try:
            async with async_playwright() as p:
                # Launch headless Chromium
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                logger.info(f"[Playwright] Navigating to: {url}")

                # Wait until network idle so JS is done
                await page.goto(url, wait_until="networkidle")

                # Grab final rendered HTML
                html = await page.content()
                await browser.close()

                return BeautifulSoup(html, "html.parser")

        except Exception as e:
            logger.error(f"Playwright render failed for {url}: {e}")
            return None


async def fetch_pdfs(
    base_url,
    url,
    visited_urls,
    session,
    output_dir,
    base_pdf=None,
    max_depth=None,
    depth=0,
    chunk_size=None,
    render_js=False,
):
    """
    Recursively crawl a website starting from ``url``, and download PDF files.

    This function performs the following steps:
      1. Fetches or renders the page at ``url`` (HTML).
         - If ``render_js=True``, uses Playwright to load JavaScript.
         - Otherwise, uses ``aiohttp`` to get the HTML.
      2. Extracts all links (<a> tags), converting them to absolute URLs.
      3. For each link:
         - If it ends in ``.pdf`` or contains ``download.php``, downloads the
         file via :func:`download_pdf_async`.
         - If the link is in the same domain as ``base_url`` and hasn't been
         visited yet, recurses into that page up to ``max_depth``, if specified.

    Parameters
    ----------
    base_url : str
        The base domain or URL to restrict recursion.
    url : str
        The current URL being processed in the recursive crawl.
    visited_urls : set
        A set of URLs already visited, to avoid duplicates.
    session : aiohttp.ClientSession
        An existing aiohttp.ClientSession
    output_dir : str
        The directory in which to save downloaded PDFs.
    base_pdf : str or None, optional
        Base URL for PDF links if different from standard usage.
        Defaults to None.
    max_depth : int or None, optional
        Maximum depth of recursion. If None, there is no depth limit.
    depth : int, optional
        Current recursion depth (used internally).
    chunk_size : int or None, optional
        If provided, PDFs are downloaded in chunks of this size (in bytes).
        If None, each PDF is downloaded at once.
    render_js : bool, optional
        Whether to use Playwright to fully render the page's JavaScript before
        scraping links. Defaults to False.

    Returns
    -------
    None
        The function uses recursion and side effects for downloading PDFs, so
        nothing is returned.

    Notes
    -----
    - If a link includes ".pdf" or "download.php", it is considered a PDF link
      and is passed to :func:`download_pdf_async`.
    - If the link is in the same domain as ``base_url`` and not yet visited,
      this function calls itself recursively.
    """
    if max_depth is not None and depth > max_depth:
        return

    if url in visited_urls:
        return
    visited_urls.add(url)

    logger.info(f"Visiting: {url}")

    try:
        soup = await fetch_and_parse(session, url, render_js=render_js)
        if not soup:
            # Not HTML or failed => skip link extraction
            return

        # Extract <a href>, convert to absolute
        links = []
        for a_tag in soup.find_all("a", href=True):
            if base_pdf is not None and ".pdf" in a_tag["href"]:
                absolute_link = urljoin(base_pdf, a_tag["href"])
            else:
                absolute_link = urljoin(url, a_tag["href"])
            links.append(absolute_link)

        tasks = []
        for link in links:
            link_lower = link.lower()

            # Skip known irrelevant extensions
            if any(ext in link_lower for ext in IGNORED_EXTENSIONS):
                continue

            # If it's a PDF or 'download.php'
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
            elif base_url in link:
                # Recurse deeper
                tasks.append(
                    fetch_pdfs(
                        base_url=base_url,
                        url=link,
                        visited_urls=visited_urls,
                        session=session,
                        output_dir=output_dir,
                        max_depth=max_depth,
                        depth=depth + 1,
                        chunk_size=chunk_size,
                        render_js=render_js,
                    )
                )

        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Error processing {url}: {e}")


async def main():
    """
    Main entry point for the async PDF scraping script.

    Steps
    -----
    1. Set up variables: ``base_url``, ``start_url``, ``base_pdf``,
        ``output_dir``, etc.
    2. Create the output directory, if needed.
    3. Set up logging to both console and a log file.
    4. Create an ``aiohttp.ClientSession`` for network requests.
    5. Call :func:`fetch_pdfs` to begin recursive scraping and PDF downloads.
    6. Close the session and log completion.

    Returns
    -------
    None
        This function completes the asynchronous crawl and downloads PDFs as
        needed.

    Notes
    -----
    - Modify the variables ``base_url``, ``start_url``, etc. to suit your
        target site.
    - Toggle ``render_js`` to True if you want to use Playwright to fully load
        JavaScript
      on each page (slower, but can capture dynamic content).
    """
    base_url = ""  # Set your base URL here
    start_url = ""  # Set your start URL here (can be the same as base_url)
    base_pdf = None  # Set your base PDF URL here (if any)
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    visited_urls = set()

    # Setup File Logger ---
    log_file_path = os.path.join(output_dir, "scrape.log")
    file_handler = logging.FileHandler(
        log_file_path, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # log all levels to file
    file_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    RENDER_JS = False  # Set to True to render dynamic JS pages
    MAX_DEPTH = None  # None for no limit
    CHUNK_SIZE = 4096  # or None

    async with aiohttp.ClientSession() as session:
        await fetch_pdfs(
            base_url=base_url,
            url=start_url,
            visited_urls=visited_urls,
            session=session,
            output_dir=output_dir,
            base_pdf=base_pdf,
            max_depth=MAX_DEPTH,  # If you want to limit recursion
            chunk_size=CHUNK_SIZE,
            render_js=RENDER_JS,
        )

    logger.info("Crawling complete.")


if __name__ == "__main__":
    asyncio.run(main())
