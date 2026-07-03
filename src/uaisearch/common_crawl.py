from collections.abc import Iterator
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from simhash import Simhash
from warcio.archiveiterator import ArchiveIterator

from uaisearch.crawler import is_dark_web
from uaisearch.models import ExtractedPage

CC_BUCKET = "commoncrawl"


def parse_wet_stream(stream, target_domains: set[str]) -> Iterator[tuple[str, str, str]]:
    for record in ArchiveIterator(stream):
        if record.rec_type != "conversion":
            continue
        url = record.rec_headers.get_header("WARC-Target-URI")
        if not url:
            continue
        if is_dark_web(url):
            continue
        domain = (urlparse(url).hostname or "").lower()
        if target_domains and domain not in target_domains:
            continue
        text = record.content_stream().read().decode("utf-8", errors="ignore")
        yield url, domain, text


def iter_wet_records(s3_key: str, target_domains: set[str]) -> Iterator[tuple[str, str, str]]:
    # ponytail: Common Crawl's bucket is public/no-sign-request, hence UNSIGNED config
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    obj = s3.get_object(Bucket=CC_BUCKET, Key=s3_key)
    yield from parse_wet_stream(obj["Body"], target_domains)


def build_page_from_wet(url: str, domain: str, text: str, crawl_date: str) -> ExtractedPage:
    # ponytail: no HTML/DOM to measure ads from, so WET content gets a neutral
    # 0.5 prior — never scored as ad-free-perfect (index_page derives domain_quality
    # = 1 - ad_ratio, so 0.0 would mint top-quality for uncleaned text)
    return ExtractedPage(
        url=url, domain=domain, title=domain, text=text,
        ad_ratio=0.5, crawl_date=crawl_date, simhash=Simhash(text).value,
    )
