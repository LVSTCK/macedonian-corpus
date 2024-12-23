import asyncio
import aiohttp
import aiofiles
import os
from bs4 import BeautifulSoup
from urllib.parse import unquote

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

pdf_list = []


async def download_pdf_async(pdf_url, session, save_path, chunk_size=None):
    """
    Download a PDF asynchronously using aiohttp and write it to disk with
    aiofiles.

    If `chunk_size` is not None, the file is downloaded in chunks.
    Otherwise, it's read entirely into memory at once.
    """
    try:
        async with session.get(pdf_url) as response:
            response.raise_for_status()

            if chunk_size:
                # Stream in chunks
                async with aiofiles.open(save_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(
                        chunk_size
                    ):
                        await f.write(chunk)
            else:
                # Download entire content at once
                data = await response.read()
                async with aiofiles.open(save_path, "wb") as f:
                    await f.write(data)

        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")


async def fetch_and_parse(session, url):
    """Fetch the HTML content at `url` asynchronously and return a
    BeautifulSoup object."""
    async with session.get(url) as response:
        response.raise_for_status()
        text = await response.text()
        return BeautifulSoup(text, "html.parser")


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
      - Download any .pdf found (chunked or not based on `chunk_size`).
      - Recursively visit child links in the same domain.
    """
    # Respect the max_depth if provided
    if max_depth is not None and depth > max_depth:
        return

    # Avoid revisiting the same page
    if url in visited_urls:
        return
    visited_urls.add(url)

    print(f"Visiting: {url}")

    try:
        soup = await fetch_and_parse(session, url)
        links = [a["href"] for a in soup.find_all("a", href=True)]

        tasks = []
        for link in links:
            link_lower = link.lower()

            # Skip links containing ignored extensions
            if any(ext in link_lower for ext in IGNORED_EXTENSIONS):
                continue

            # If it's a PDF link, download it
            if ".pdf" in link_lower:
                raw_filename = link.split("/")[-1].split("?")[0].split("#")[0]
                decoded_filename = unquote(raw_filename)
                save_path = os.path.join(output_dir, decoded_filename)
                print("Downloading PDF: ", link)
                tasks.append(
                    download_pdf_async(link, session, save_path, chunk_size)
                )

            # Otherwise, if it's in the same domain, recurse deeper
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

        # Gather all download + sub-crawl tasks
        await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Error processing {url}: {e}")


async def main():
    base_url = ""
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    visited_urls = set()

    CHUNK_SIZE = None  # or None

    async with aiohttp.ClientSession() as session:
        await fetch_pdfs(
            base_url=base_url,
            url=base_url,
            visited_urls=visited_urls,
            session=session,
            output_dir=output_dir,
            max_depth=2,  # If you want to limit recursion depth
            chunk_size=CHUNK_SIZE,
        )

    print("Crawling complete.")


if __name__ == "__main__":
    asyncio.run(main())
