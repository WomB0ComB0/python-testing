#!/usr/bin/python
# -*- python -*-

from __future__ import annotations

# Standard library imports
import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Final,
    List,
    Optional,
    Set,
    TypedDict,
    Union,
)

# Third-party imports
import aiofiles
import tiktoken
from bs4 import BeautifulSoup, Tag
from dotenv import find_dotenv, load_dotenv
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enumeration for different content types."""

    ARTICLE = auto()
    BLOG_POST = auto()
    NEWS = auto()
    PRODUCT = auto()
    DOCUMENTATION = auto()
    SOCIAL_MEDIA = auto()
    UNKNOWN = auto()


class ProcessingStatus(Enum):
    """Enumeration for processing status."""

    PENDING = auto()
    SCRAPING = auto()
    ANALYZING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass(frozen=True, slots=True)
class ScrapingConfig:
    """Configuration for web scraping operations."""

    timeout: int = 30000
    wait_for_selector: Optional[str] = None
    screenshot: bool = False
    pdf: bool = False
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    viewport_width: int = 1920
    viewport_height: int = 1080


class ContentMetadata(TypedDict, total=False):
    """Type definition for content metadata."""

    title: str
    description: str
    keywords: List[str]
    author: str
    published_date: str
    word_count: int
    reading_time: int
    language: str


class AnalysisResult(BaseModel):
    """Pydantic model for analysis results."""

    url: str
    content_type: ContentType
    metadata: ContentMetadata
    summary: str = Field(description="Brief summary of the content")
    key_points: List[str] = Field(description="Main points extracted from content")
    sentiment: str = Field(
        description="Overall sentiment: positive, negative, or neutral"
    )
    readability_score: float = Field(
        ge=0, le=100, description="Readability score (0-100)"
    )
    topics: List[str] = Field(description="Main topics covered")
    processing_time: float = Field(description="Time taken to process in seconds")


@dataclass
class WebContentAnalyzer:
    """Main class for web content analysis operations."""

    config: ScrapingConfig = ScrapingConfig()
    max_concurrent: int = 5
    cache_dir: Path = Path("./cache")
    results_dir: Path = Path("./results")

    def __post_init__(self):
        """Initialize directories and setup."""
        self.cache_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    @cached_property
    def tokenizer(self):
        """Get tiktoken encoder for token counting."""
        try:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def _generate_cache_key(self, url: str) -> str:
        """Generate a unique cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"{cache_key}.json"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((OSError, TimeoutError)),
    )
    async def _scrape_with_playwright(self, url: str) -> Dict[str, Any]:
        """Scrape content using Playwright with retry logic."""
        async with self._semaphore:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=self.config.user_agent,
                        viewport={
                            "width": self.config.viewport_width,
                            "height": self.config.viewport_height,
                        },
                    )
                    page = await context.new_page()

                    await page.goto(
                        url, timeout=self.config.timeout, wait_until="domcontentloaded"
                    )

                    if self.config.wait_for_selector:
                        await page.wait_for_selector(
                            self.config.wait_for_selector, timeout=10000
                        )

                    # Extract content
                    content = await page.content()
                    title = await page.title()

                    # Optional screenshot
                    screenshot_data = None
                    if self.config.screenshot:
                        screenshot_data = await page.screenshot()

                    return {
                        "html": content,
                        "title": title,
                        "url": url,
                        "screenshot": screenshot_data,
                        "timestamp": time.time(),
                    }

                finally:
                    await browser.close()

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> ContentMetadata:
        """Extract metadata from BeautifulSoup object."""
        metadata: ContentMetadata = {}

        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()

        # Meta description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag["content"].strip()

        # Keywords
        keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            metadata["keywords"] = [
                k.strip() for k in keywords_tag["content"].split(",")
            ]

        # Author
        author_tag = soup.find("meta", attrs={"name": "author"}) or soup.find(
            "meta", attrs={"property": "article:author"}
        )
        if author_tag and author_tag.get("content"):
            metadata["author"] = author_tag["content"].strip()

        # Published date
        date_tag = soup.find(
            "meta", attrs={"property": "article:published_time"}
        ) or soup.find("time")
        if date_tag:
            if date_tag.get("datetime"):
                metadata["published_date"] = date_tag["datetime"]
            elif date_tag.get("content"):
                metadata["published_date"] = date_tag["content"]

        # Word count and reading time
        text_content = soup.get_text()
        words = len(re.findall(r"\b\w+\b", text_content))
        metadata["word_count"] = words
        metadata["reading_time"] = max(
            1, words // 200
        )  # Assuming 200 WPM reading speed

        # Language
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            metadata["language"] = html_tag["lang"]

        return metadata

    def _classify_content_type(self, soup: BeautifulSoup, url: str) -> ContentType:
        """Classify the type of content based on HTML structure and URL."""
        url_lower = url.lower()

        # Check URL patterns
        if any(pattern in url_lower for pattern in ["blog", "post", "article"]):
            return ContentType.BLOG_POST
        elif any(pattern in url_lower for pattern in ["news", "press", "media"]):
            return ContentType.NEWS
        elif any(
            pattern in url_lower for pattern in ["product", "shop", "buy", "store"]
        ):
            return ContentType.PRODUCT
        elif any(
            pattern in url_lower
            for pattern in ["docs", "documentation", "guide", "manual"]
        ):
            return ContentType.DOCUMENTATION

        # Check HTML structure
        if soup.find("article"):
            return ContentType.ARTICLE
        elif soup.find("meta", attrs={"property": "og:type", "content": "article"}):
            return ContentType.ARTICLE

        return ContentType.UNKNOWN

    def _calculate_readability(self, text: str) -> float:
        """Calculate a simple readability score (Flesch Reading Ease approximation)."""
        sentences = len(re.split(r"[.!?]+", text))
        words = len(re.findall(r"\b\w+\b", text))
        syllables = sum(
            max(1, len(re.findall(r"[aeiouyAEIOUY]", word)))
            for word in re.findall(r"\b\w+\b", text)
        )

        if sentences == 0 or words == 0:
            return 0.0

        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text using simple heuristics."""
        # Find sentences with emphasis (bold, headers, etc.)
        sentences = re.split(r"[.!?]+", text)
        key_points = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                # Simple scoring based on position and length
                if any(
                    keyword in sentence.lower()
                    for keyword in ["important", "key", "main", "crucial", "essential"]
                ):
                    key_points.append(sentence)
                elif len(key_points) < 5 and len(sentence) > 50:
                    key_points.append(sentence)

        return key_points[:5]  # Return top 5 key points

    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis using keyword matching."""
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "positive",
            "success",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "negative",
            "problem",
            "issue",
            "fail",
        ]

        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _extract_topics(self, text: str, metadata: ContentMetadata) -> List[str]:
        """Extract main topics from text and metadata."""
        topics = set()

        # From keywords if available
        if "keywords" in metadata:
            topics.update(metadata["keywords"])

        # Simple topic extraction from text
        common_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        word_freq = {}
        for word in common_words:
            word_lower = word.lower()
            if len(word_lower) > 3 and word_lower not in [
                "this",
                "that",
                "with",
                "from",
                "they",
                "have",
                "been",
            ]:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top topics by frequency
        top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        topics.update([topic[0] for topic in top_topics])

        return list(topics)[:10]

    async def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load content from cache if available and fresh."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            async with aiofiles.open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.loads(await f.read())

            # Check if cache is still fresh (24 hours)
            if time.time() - cached_data.get("timestamp", 0) < 86400:
                logger.info(
                    format("Loading from cache: {cache_key}", cache_key=cache_key)
                )
                return cached_data

        except Exception as e:
            logger.warning(
                format("Error loading cache {cache_key}: {e}", cache_key=cache_key, e=e)
            )

        return None

    async def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save scraped data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
            logger.info(format("Saved to cache: {cache_key}", cache_key=cache_key))
        except Exception as e:
            logger.error(
                format("Error saving cache {cache_key}: {e}", cache_key=cache_key, e=e)
            )

    async def analyze_url(self, url: str) -> AnalysisResult:
        """Analyze a single URL and return comprehensive results."""
        start_time = time.time()
        cache_key = self._generate_cache_key(url)

        try:
            # Try to load from cache first
            cached_data = await self._load_from_cache(cache_key)
            if cached_data:
                scraped_data = cached_data
            else:
                # Scrape fresh content
                logger.info(format("Scraping URL: {url}", url=url))
                scraped_data = await self._scrape_with_playwright(url)
                await self._save_to_cache(cache_key, scraped_data)

            # Parse HTML
            soup = BeautifulSoup(scraped_data["html"], "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            # Extract text content
            text_content = soup.get_text()
            clean_text = " ".join(text_content.split())  # Clean whitespace

            # Extract metadata
            metadata = self._extract_metadata(soup, url)

            # Classify content type
            content_type = self._classify_content_type(soup, url)

            # Perform analysis
            key_points = self._extract_key_points(clean_text)
            sentiment = self._analyze_sentiment(clean_text)
            readability_score = self._calculate_readability(clean_text)
            topics = self._extract_topics(clean_text, metadata)

            # Create summary (first few sentences)
            sentences = re.split(r"[.!?]+", clean_text)
            summary_sentences = [
                s.strip() for s in sentences[:3] if len(s.strip()) > 20
            ]
            summary = (
                ". ".join(summary_sentences) + "."
                if summary_sentences
                else "No summary available."
            )

            processing_time = time.time() - start_time

            result = AnalysisResult(
                url=url,
                content_type=content_type,
                metadata=metadata,
                summary=summary,
                key_points=key_points,
                sentiment=sentiment,
                readability_score=readability_score,
                topics=topics,
                processing_time=processing_time,
            )

            logger.info(
                format(
                    "Analysis completed for {url} in {processing_time:.2f}s",
                    url=url,
                    processing_time=processing_time,
                )
            )
            return result

        except Exception as e:
            logger.error(format("Error analyzing {url}: {e}", url=url, e=e))
            raise

    async def analyze_urls(self, urls: List[str]) -> List[AnalysisResult]:
        """Analyze multiple URLs concurrently."""
        logger.info(format("Starting analysis of {len} URLs", len=len(urls)))

        tasks = [self.analyze_url(url) for url in urls]
        results = []

        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(format("Failed to analyze URL: %s - %s", task, e))

        logger.info(format("Completed analysis of %s/%s URLs", len(results), len(urls)))
        return results

    async def save_results(
        self, results: List[AnalysisResult], filename: str = None
    ) -> Path:
        """Save analysis results to JSON file."""
        if filename is None:
            filename = f"analysis_results_{int(time.time())}.json"

        output_path = self.results_dir / filename

        # Convert to dict for JSON serialization
        results_data = {
            "timestamp": time.time(),
            "total_analyzed": len(results),
            "results": [result.model_dump() for result in results],
        }

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(
                json.dumps(results_data, ensure_ascii=False, indent=2, default=str)
            )

        logger.info(format("Results saved to %s", output_path))
        return output_path

    def get_summary_stats(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results."""
        if not results:
            return {}

        total_results = len(results)
        content_types = {}
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        avg_readability = sum(r.readability_score for r in results) / total_results
        avg_processing_time = sum(r.processing_time for r in results) / total_results

        for result in results:
            # Count content types
            content_type = result.content_type.name
            content_types[content_type] = content_types.get(content_type, 0) + 1

            # Count sentiments
            if result.sentiment in sentiments:
                sentiments[result.sentiment] += 1

        # Most common topics
        all_topics = []
        for result in results:
            all_topics.extend(result.topics)

        topic_freq = {}
        for topic in all_topics:
            topic_freq[topic] = topic_freq.get(topic, 0) + 1

        top_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_analyzed": total_results,
            "content_types": content_types,
            "sentiment_distribution": sentiments,
            "average_readability_score": round(avg_readability, 2),
            "average_processing_time": round(avg_processing_time, 2),
            "top_topics": dict(top_topics),
        }


# Example usage and CLI interface
async def main():
    """Example usage of the WebContentAnalyzer."""
    # Sample URLs for testing
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://jsonplaceholder.typicode.com/posts/1",
    ]

    # Initialize analyzer
    config = ScrapingConfig(
        timeout=30000,
        screenshot=False,  # Set to True if you want screenshots
        user_agent="WebContentAnalyzer/1.0",
    )

    analyzer = WebContentAnalyzer(config=config, max_concurrent=3)

    try:
        print("Starting web content analysis...")

        # Analyze URLs
        results = await analyzer.analyze_urls(test_urls)

        # Save results
        output_path = await analyzer.save_results(results)
        print(f"Results saved to: {output_path}")

        # Print summary
        stats = analyzer.get_summary_stats(results)
        print("\nSummary Statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

        # Print individual results
        print("\nDetailed Results:")
        for result in results:
            print(f"\nURL: {result.url}")
            print(f"Content Type: {result.content_type.name}")
            print(f"Title: {result.metadata.get('title', 'N/A')}")
            print(f"Word Count: {result.metadata.get('word_count', 'N/A')}")
            print(f"Sentiment: {result.sentiment}")
            print(f"Readability Score: {result.readability_score:.1f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Summary: {result.summary[:200]}...")
            print(f"Topics: {', '.join(result.topics[:5])}")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(format("Analysis failed: {e}", e=e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
