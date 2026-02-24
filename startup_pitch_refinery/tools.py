import json
import io
import os
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from linkup import LinkupClient
from pptx import Presentation


class MarketSearchTool:
    """Wrapper around Linkup search with graceful fallback when key is missing."""

    def __init__(self) -> None:
        self._client = None
        allowed_env = os.getenv("LINKUP_ALLOWED_DOMAINS", "").strip()
        blocked_env = os.getenv("LINKUP_BLOCKED_DOMAINS", "").strip()
        default_blocked = {
            "github.com",
            "marketreportsworld.com",
            "verifiedmarketreports.com",
            "futuredatastats.com",
            "dataintelo.com",
            "marketintelo.com",
            "htfmarketinsights.com",
        }
        self._allowed_domains = {
            d.strip().lower() for d in allowed_env.split(",") if d.strip()
        }
        blocked_domains = {d.strip().lower() for d in blocked_env.split(",") if d.strip()}
        self._blocked_domains = blocked_domains or default_blocked
        try:
            self._client = LinkupClient()
        except Exception:  # noqa: BLE001
            self._client = None

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:  # noqa: BLE001
            return ""

    def _is_domain_allowed(self, domain: str) -> bool:
        if not domain:
            return False
        if self._allowed_domains:
            return domain in self._allowed_domains
        return domain not in self._blocked_domains

    def search(self, query: str) -> Dict[str, Any]:
        if self._client is None:
            return {
                "status": "error",
                "query": query,
                "results_json": "[]",
                "results": [],
                "sources": [],
                "dropped_sources": 0,
                "error": "Set LINKUP_API_KEY to enable live web search.",
            }
        try:
            response = self._client.search(
                query=query,
                depth="standard",
                output_type="searchResults",
            )
            raw_results = getattr(response, "results", [])
            compact = []
            seen_urls = set()
            dropped_count = 0
            for r in raw_results:
                if isinstance(r, dict):
                    title = r.get("name", "")
                    url = r.get("url", "")
                    content = r.get("content", "")
                else:
                    title = getattr(r, "name", "")
                    url = getattr(r, "url", "")
                    content = getattr(r, "content", "")

                if not url or url in seen_urls:
                    continue
                domain = self._extract_domain(url)
                if not self._is_domain_allowed(domain):
                    dropped_count += 1
                    continue
                seen_urls.add(url)
                compact.append(
                    {
                        "title": title,
                        "url": url,
                        "domain": domain,
                        "content": str(content)[:400],
                    }
                )
            sources = [item["url"] for item in compact if item.get("url")]
            return {
                "status": "ok",
                "query": query,
                "results_json": json.dumps(compact, indent=2),
                "results": compact,
                "sources": sources,
                "dropped_sources": dropped_count,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "query": query,
                "results_json": "[]",
                "results": [],
                "sources": [],
                "dropped_sources": 0,
                "error": f"Market search failed: {exc}",
            }


class BusinessCalcTool:
    """Python REPL helper for simple financial projections."""

    def run(self, code: str) -> str:
        try:
            buf = io.StringIO()
            globals_dict = {"__builtins__": {"print": print, "range": range, "len": len}}
            with redirect_stdout(buf):
                exec(code, globals_dict, {})
            return buf.getvalue().strip()
        except Exception as exc:  # noqa: BLE001
            return f"Python calc failed: {exc}"


class GoogleTrendsTool:
    """Google Trends wrapper via pytrends."""

    def __init__(self) -> None:
        self.hl = os.getenv("GOOGLE_TRENDS_HL", "en-US").strip()
        self.tz = int(os.getenv("GOOGLE_TRENDS_TZ", "360").strip() or "360")
        self.geo = os.getenv("GOOGLE_TRENDS_GEO", "US").strip()
        self.timeframe = os.getenv("GOOGLE_TRENDS_TIMEFRAME", "today 12-m").strip()
        self.connect_timeout = int(os.getenv("GOOGLE_TRENDS_CONNECT_TIMEOUT", "6").strip())
        self.read_timeout = int(os.getenv("GOOGLE_TRENDS_READ_TIMEOUT", "12").strip())
        self.retries = int(os.getenv("GOOGLE_TRENDS_RETRIES", "2").strip())
        self.backoff_factor = float(os.getenv("GOOGLE_TRENDS_BACKOFF_FACTOR", "0.4").strip())
        self.max_keywords = int(os.getenv("GOOGLE_TRENDS_MAX_KEYWORDS", "5").strip())
        self._init_error = ""
        try:
            from pytrends.request import TrendReq

            self._client = TrendReq(
                hl=self.hl,
                tz=self.tz,
                timeout=(self.connect_timeout, self.read_timeout),
                retries=self.retries,
                backoff_factor=self.backoff_factor,
            )
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self._init_error = str(exc)

    def fetch(self, keywords: List[str]) -> Dict[str, Any]:
        clean_keywords = [k.strip() for k in keywords if k and k.strip()][: self.max_keywords]
        if not clean_keywords:
            return {
                "status": "error",
                "keywords": [],
                "data": {},
                "error": "No keywords provided for trends lookup.",
            }
        if self._client is None:
            return {
                "status": "error",
                "keywords": clean_keywords,
                "data": {},
                "error": (
                    "Google Trends unavailable. Ensure pytrends is installed in the same "
                    f"Python environment. Details: {self._init_error or 'import failed'}"
                ),
            }

        per_keyword: Dict[str, Any] = {}
        failures: List[str] = []
        for kw in clean_keywords:
            attempts = max(1, self.retries + 1)
            last_error = ""
            for attempt in range(attempts):
                try:
                    self._client.build_payload([kw], timeframe=self.timeframe, geo=self.geo)
                    iot = self._client.interest_over_time()
                    related = self._client.related_queries() or {}

                    summary = {
                        "average_interest": 0.0,
                        "latest_interest": 0,
                        "peak_interest": 0,
                        "top_queries": [],
                        "rising_queries": [],
                    }
                    if not iot.empty and kw in iot.columns:
                        values = [int(v) for v in iot[kw].tolist()]
                        if values:
                            summary["average_interest"] = round(sum(values) / len(values), 2)
                            summary["latest_interest"] = values[-1]
                            summary["peak_interest"] = max(values)

                    kw_related = related.get(kw, {})
                    top_df = kw_related.get("top")
                    if top_df is not None and not top_df.empty and "query" in top_df.columns:
                        summary["top_queries"] = [
                            str(q) for q in top_df["query"].head(5).tolist()
                        ]
                    rising_df = kw_related.get("rising")
                    if (
                        rising_df is not None
                        and not rising_df.empty
                        and "query" in rising_df.columns
                    ):
                        summary["rising_queries"] = [
                            str(q) for q in rising_df["query"].head(5).tolist()
                        ]
                    per_keyword[kw] = summary
                    last_error = ""
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if attempt < attempts - 1:
                        sleep_s = self.backoff_factor * (2**attempt)
                        time.sleep(max(0.2, sleep_s))
            if last_error:
                failures.append(f"{kw}: {last_error}")

        if not per_keyword:
            return {
                "status": "error",
                "keywords": clean_keywords,
                "data": {},
                "error": "; ".join(failures)[:500] or "Google Trends request failed.",
            }

        status = "ok" if not failures else "partial"
        return {
            "status": status,
            "keywords": clean_keywords,
            "data": per_keyword,
            "error": "; ".join(failures)[:500] if failures else "",
        }


class ScenarioAnalysisTool:
    """Deterministic scenario projections for startup plans."""

    def run(
        self,
        users_year1: int = 8000,
        arpu_monthly: float = 12.0,
        gross_margin: float = 0.72,
    ) -> Dict[str, Any]:
        scenarios = {
            "conservative": {"users_multiplier": 0.7, "arpu_multiplier": 0.9},
            "base": {"users_multiplier": 1.0, "arpu_multiplier": 1.0},
            "aggressive": {"users_multiplier": 1.35, "arpu_multiplier": 1.15},
        }
        output: Dict[str, Any] = {}
        for name, mult in scenarios.items():
            users = int(round(users_year1 * mult["users_multiplier"]))
            arpu = arpu_monthly * mult["arpu_multiplier"]
            revenue = users * arpu * 12
            gross_profit = revenue * gross_margin
            output[name] = {
                "users": users,
                "arpu_monthly": round(arpu, 2),
                "annual_revenue": round(revenue, 2),
                "gross_profit": round(gross_profit, 2),
                "gross_margin": gross_margin,
            }
        return output


def generate_pitch_deck(slides: Dict[str, str], output_path: str) -> str:
    """Create a .pptx deck from text content."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()

    title_layout = prs.slide_layouts[0]
    body_layout = prs.slide_layouts[1]

    title_slide = prs.slides.add_slide(title_layout)
    title_slide.shapes.title.text = slides.get("title", "Startup Pitch")
    title_slide.placeholders[1].text = slides.get("subtitle", "AI Startup Pitch Refinery")

    ordered_sections: List[str] = [
        "problem",
        "solution",
        "market",
        "business_model",
        "competitive_advantage",
        "financials",
    ]

    headers = {
        "problem": "Problem",
        "solution": "Solution",
        "market": "Market",
        "business_model": "Business Model",
        "competitive_advantage": "Competitive Advantage",
        "financials": "Financials",
    }

    for key in ordered_sections:
        slide = prs.slides.add_slide(body_layout)
        slide.shapes.title.text = headers[key]
        slide.placeholders[1].text = slides.get(key, "TBD")

    prs.save(path)
    return str(path)
