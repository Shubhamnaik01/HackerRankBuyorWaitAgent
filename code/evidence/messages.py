from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from ..models import Message
from .models import EvidenceFact


_FINANCIAL_MARKERS = (
    "salary", "payroll", "income", "payout", "invoice", "refund", "bonus",
    "commission", "employment", "contract", "rent", "lease", "debit",
    "credit", "transfer", "investment", "portfolio", "prize", "bill",
    "gaji", "penggajian", "pendapatan", "pembayaran", "faktur",
    "pengembalian dana", "sewa", "transfer", "investasi", "hadiah",
)
_MONEY = re.compile(r"\b(INR|IDR|USD|EUR|ZAR)\s*([0-9][0-9,.]*)\b", re.IGNORECASE)
_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_PERCENT = re.compile(r"(?:by|sebesar)\s+(\d+(?:\.\d+)?)%", re.IGNORECASE)


def is_financially_relevant(message: Message) -> bool:
    text = message.message_text.casefold()
    return any(marker in text for marker in _FINANCIAL_MARKERS)


def _first_date(text: str) -> date | None:
    match = _DATE.search(text)
    return date.fromisoformat(match.group(1)) if match else None


def _money_values(text: str) -> list[tuple[str, Decimal]]:
    return [
        (match.group(1).upper(), Decimal(match.group(2).replace(",", "")))
        for match in _MONEY.finditer(text)
    ]


def parse_message_deterministically(message: Message) -> tuple[list[EvidenceFact], bool]:
    """Extract facts from the finite, repetitive evidence templates.

    The second return value says the relevant message was fully classified, so
    callers do not spend a model call on known non-cash or pending evidence.
    Free-form relevant text that is not recognized can be sent to the tightly
    constrained model adapter.
    """
    text = message.message_text
    lower = text.casefold()
    source = message.message_id
    related = message.related_event_id

    if any(phrase in lower for phrase in (
        "transfer between your two accounts", "transfer antara dua rekening anda",
    )):
        return [EvidenceFact("internal_transfer", source, related)], True

    if any(phrase in lower for phrase in (
        "has not reached your account", "belum masuk ke rekening", "still processing",
        "still pending", "masih tertunda", "not withdrawable", "belum dapat ditarik",
        "not been credited", "belum dikreditkan", "no cash proceeds", "tidak ada transaksi tunai",
        "not been sold", "belum dijual", "not been approved", "belum disetujui",
    )) and not any(phrase in lower for phrase in (
        "regular salary", "monthly pay", "monthly salary", "gaji bulanan", "gaji rutin",
        "confirmed base salary", "confirmed salary", "remaining confirmed monthly salary",
        "gaji pokok", "sisa gaji bulanan",
    )):
        category = "unconfirmed_income" if any(
            marker in lower for marker in ("payout", "weekly earnings", "penghasilan mingguan")
        ) else None
        return [EvidenceFact("exclude_from_cash", source, related, category=category)], True

    if any(phrase in lower for phrase in (
        "previous debit attempt failed", "debit sebelumnya gagal",
        "another debit will be attempted", "debit akan dicoba lagi",
    )):
        return [EvidenceFact("failed_debit_due", source, related)], True

    if "rent" in lower or "lease" in lower or "sewa" in lower:
        percent = _PERCENT.search(text)
        if percent and any(marker in lower for marker in ("increase", "increases", "naik")):
            multiplier = Decimal("1") + Decimal(percent.group(1)) / Decimal("100")
            return [EvidenceFact(
                "recurring_expense_multiplier", source, related, multiplier=multiplier,
                category="rent", scope="recurring",
            )], True

    money = _money_values(text)
    effective = _first_date(text)

    if any(phrase in lower for phrase in (
        "client approved an invoice payment", "client approved a payment", "klien menyetujui pembayaran faktur",
    )) and money and effective:
        currency, amount = money[0]
        return [EvidenceFact(
            "confirmed_one_time_income", source, related, amount, currency, effective, scope="one_time"
        )], True

    remaining_job = any(phrase in lower for phrase in (
        "remaining confirmed monthly salary", "sisa gaji bulanan yang dikonfirmasi",
    ))
    ended = any(phrase in lower for phrase in (
        "your employment has ended", "current seasonal contract has ended",
        "kontrak musiman saat ini telah berakhir", "employment record has ended",
        "sumber pendapatan kerja rumah tangga telah berakhir",
    ))
    if ended and not remaining_job:
        return [EvidenceFact("employment_end", source, related, effective_date=effective)], True

    salary_context = any(marker in lower for marker in (
        "salary", "payroll", "monthly pay", "gaji", "penggajian",
    ))
    if salary_context and money:
        currency, amount = money[0]
        next_only = any(phrase in lower for phrase in (
            "next salary is reduced", "temporary monthly pay", "temporary monthly salary",
            "gaji bulanan sementara", "next payroll", "penggajian berikutnya",
        )) and not any(phrase in lower for phrase in ("first salary", "gaji pertama", "regular salary", "gaji rutin"))
        scope = "next_only" if next_only else "recurring"
        return [EvidenceFact(
            "salary_change", source, related, amount, currency, effective, scope=scope
        )], True

    if salary_context and effective and any(phrase in lower for phrase in (
        "salary is now expected", "confirmed salary is now expected",
        "gaji kini diharapkan", "replaces the payroll date", "menggantikan tanggal",
    )):
        return [EvidenceFact("salary_delay", source, related, effective_date=effective, scope="recurring")], True

    # Known informational payroll templates contain no usable amount. Image
    # evidence or structured rows may still supply it.
    if salary_context and any(phrase in lower for phrase in (
        "one-time adjustment", "penyesuaian satu kali", "quarterly bonus", "bonus kuartalan",
    )):
        return [], True

    if any(phrase in lower for phrase in (
        "claim is now closed", "no further scheduled payments", "klaim sekarang ditutup",
        "tidak ada pembayaran terjadwal lagi",
    )):
        return [EvidenceFact("exclude_from_cash", source, related)], True

    return [], False
