"""Point-in-time security-code aliases for stable internal instruments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from cosmosalpha.data.models import canonical_instrument_code

INSTRUMENT_CODE_HISTORY_SCHEMA_VERSION = "instrument-code-history/1.0"


@dataclass(frozen=True, slots=True)
class InstrumentCodeChange:
    """One exchange-confirmed security-code change."""

    historical_instrument: str
    canonical_instrument: str
    effective_date: str
    evidence_url: str

    def __post_init__(self) -> None:
        """Validate the versioned alias record."""
        historical = canonical_instrument_code(self.historical_instrument)
        canonical = canonical_instrument_code(self.canonical_instrument)
        if historical == canonical:
            raise ValueError("historical and canonical instruments must differ")
        parsed = date.fromisoformat(self.effective_date)
        if parsed.isoformat() != self.effective_date:
            raise ValueError("effective_date must use YYYY-MM-DD")
        if not self.evidence_url.startswith("https://"):
            raise ValueError("evidence_url must be an HTTPS URL")
        object.__setattr__(self, "historical_instrument", historical)
        object.__setattr__(self, "canonical_instrument", canonical)


INSTRUMENT_CODE_CHANGES: tuple[InstrumentCodeChange, ...] = (
    InstrumentCodeChange(
        historical_instrument="sz300114",
        canonical_instrument="sz302132",
        effective_date="2025-02-17",
        evidence_url=(
            "https://disc.static.szse.cn/disc/disk03/finalpage/"
            "2025-04-29/8a7947ed-01cb-4b68-936e-e44974f1f734.PDF"
        ),
    ),
    InstrumentCodeChange(
        historical_instrument="sz000022",
        canonical_instrument="sz001872",
        effective_date="2018-12-26",
        evidence_url=(
            "https://disc.static.szse.cn/disc/disk02/finalpage/"
            "2019-11-01/c1b3dbe4-c4e5-477f-9681-5a7d1cb59996.PDF"
        ),
    ),
    InstrumentCodeChange(
        historical_instrument="sz000043",
        canonical_instrument="sz001914",
        effective_date="2019-12-16",
        evidence_url=(
            "https://www.szse.cn/disclosure/notice/general/" "t20191211_572534.html"
        ),
    ),
    InstrumentCodeChange(
        historical_instrument="sh601313",
        canonical_instrument="sh601360",
        effective_date="2018-02-28",
        evidence_url=(
            "https://www.sse.com.cn/disclosure/fund/announcement/c/"
            "2018-03-28/510290_2017_n.pdf"
        ),
    ),
)

_CHANGE_BY_ALIAS = {
    alias: change
    for change in INSTRUMENT_CODE_CHANGES
    for alias in (change.historical_instrument, change.canonical_instrument)
}


def canonical_instrument_from_tushare(ts_code: str) -> str:
    """Map a historical or backfilled Tushare code to one stable instrument."""
    code, separator, exchange = ts_code.strip().upper().partition(".")
    if not separator or not code or exchange not in {"SH", "SZ", "BJ"}:
        raise ValueError(f"Unsupported Tushare stock code: {ts_code!r}")
    instrument = canonical_instrument_code(f"{exchange.lower()}{code}")
    change = _CHANGE_BY_ALIAS.get(instrument)
    return change.canonical_instrument if change is not None else instrument


def tushare_instrument_for_date(instrument: str, on_date: str) -> str:
    """Return the security code valid on a historical date for source queries."""
    canonical = canonical_instrument_code(instrument)
    parsed_date = date.fromisoformat(on_date)
    if parsed_date.isoformat() != on_date:
        raise ValueError("on_date must use YYYY-MM-DD")
    change = _CHANGE_BY_ALIAS.get(canonical)
    if change is not None:
        canonical = (
            change.historical_instrument
            if on_date < change.effective_date
            else change.canonical_instrument
        )
    exchange = canonical[:2].upper()
    return f"{canonical[2:]}.{exchange}"


def instrument_code_history_manifest() -> dict[str, object]:
    """Return auditable alias metadata for data-build manifests."""
    return {
        "schema_version": INSTRUMENT_CODE_HISTORY_SCHEMA_VERSION,
        "changes": [asdict(change) for change in INSTRUMENT_CODE_CHANGES],
    }
