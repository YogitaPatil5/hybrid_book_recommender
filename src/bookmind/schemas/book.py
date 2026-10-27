"""Book-related response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class BuyLinks(BaseModel):
    """Retailer search-result URLs for a book (never fabricated product IDs)."""

    model_config = ConfigDict(frozen=True)

    amazon: HttpUrl
    flipkart: HttpUrl
    google_books: HttpUrl


class Book(BaseModel):
    """A single search / recommendation result."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., min_length=1)
    author: str = "Unknown"
    genre: str = "Other"
    description: str = ""
    image_url: str | None = Field(
        default=None, description="Cover URL, or null when no cover is available"
    )
    avg_rating: float | None = Field(default=None, ge=0, le=5)
    published_year: int | None = Field(default=None, ge=0, le=2100)
    score: float | None = Field(default=None, description="Cosine similarity, higher is closer")
    buy_links: BuyLinks


class PopularBook(BaseModel):
    """Entry on the home page 'popular books' list."""

    model_config = ConfigDict(frozen=True)

    title: str
    author: str = "Unknown"
    image_url: str | None = None
    avg_rating: float | None = None
    num_ratings: int | None = Field(default=None, ge=0)
    genre: str | None = None
    buy_links: BuyLinks
