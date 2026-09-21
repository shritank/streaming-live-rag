"""Seeded builder for the dev corpus (Task 1's first milestone).

Emits ~40 short synthetic policy/venue/travel/catering documents as Markdown
with numbered sections, plus qrels.jsonl mapping a query to its relevant
`Doc_ID §Section`. Everything is derived from a seeded RNG, so:

  python -m fixtures.build_dev_corpus --seed 1 --out fixtures/dev_corpus

is byte-reproducible, and a different seed produces a fully re-skinned corpus
(different cities, vendors, numbers) for testing that nothing is hardcoded.

The dev corpus is a FIXTURE, not knowledge. The engine must behave identically
on the real corpus, which will be dropped into data/corpus/ later.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

CITIES = ["Pune", "Bengaluru", "Hyderabad", "Chennai", "Jaipur", "Kochi", "Indore", "Nagpur"]
VENUE_NAMES = ["Venue A", "Venue B", "Venue C", "Riverside Hall", "Summit Centre", "Lakeview Rooms"]
VENDORS = ["Green Leaf Catering", "Spice Route Foods", "Metro Banquets", "Harvest Kitchen"]
DEPARTMENTS = ["Sales", "Engineering", "Customer Success", "Field Operations"]
CURRENCIES = ["USD", "EUR", "SGD", "GBP"]


def _doc(doc_id: str, title: str, sections: list[tuple[str, str]]) -> str:
    lines = [f"---", f"doc_id: {doc_id}", f"title: {title}", "---", "", f"# {title}", ""]
    for i, (heading, body) in enumerate(sections, start=1):
        lines.append(f"## §{i} {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def build_venue_doc(rng, doc_id, city, venue):
    cap = rng.choice([25, 30, 40, 60, 80, 120])
    rooms = rng.randint(2, 6)
    notice = rng.choice([48, 72, 96])
    sections = [
        ("Overview",
         f"{venue} is a conference facility located in {city}. It is used for customer workshops, "
         f"training sessions and internal offsites."),
        ("Capacity and layout",
         f"{venue} supports a maximum workshop capacity of {cap} attendees in theatre layout, across "
         f"{rooms} breakout rooms. Classroom layout reduces capacity by approximately 30 percent."),
        ("Booking and notice",
         f"Reservations for {venue} must be confirmed at least {notice} hours in advance. "
         f"Provisional holds expire automatically after 7 days without written confirmation."),
        ("Facilities",
         f"The venue provides projection, wired and wireless microphones, and on-site technical support "
         f"during business hours. Accessible entrances and step-free access are available at all levels."),
    ]
    return _doc(doc_id, f"{venue} — {city} facility profile", sections), [
        (f"{venue} workshop capacity in {city}", f"{doc_id} §2", "lexical"),
        (f"booking notice period for {venue} in {city}", f"{doc_id} §3", "lexical"),
        (f"how many people fit in {venue} {city}", f"{doc_id} §2", "paraphrase"),
        (f"how far ahead must I reserve {venue} {city}", f"{doc_id} §3", "paraphrase"),
    ]


def build_cancellation_doc(rng, doc_id, city):
    full = rng.choice([72, 96, 120])
    partial = rng.choice([24, 48])
    pct = rng.choice([25, 50, 75])
    sections = [
        ("Scope",
         f"This policy governs cancellation and refund of venue reservations made in {city}."),
        ("Full refund window",
         f"A booking cancelled more than {full} hours before the scheduled start date receives a full refund "
         f"of all fees paid, excluding non-refundable third-party deposits."),
        ("Partial refund window",
         f"A booking cancelled between {partial} and {full} hours before the start date receives a "
         f"{pct} percent refund. Cancellations inside {partial} hours are non-refundable."),
        ("Rescheduling",
         f"A reservation may be rescheduled once at no charge if the request is made more than {full} hours "
         f"in advance. Subsequent reschedules are treated as a cancellation and rebooking."),
    ]
    return _doc(doc_id, f"Cancellation and refund policy — {city}", sections), [
        (f"cancellation terms and refund policies {city}", f"{doc_id} §2", "lexical"),
        (f"partial refund window for a {city} booking", f"{doc_id} §3", "lexical"),
        (f"can I get my money back if I call off the {city} event", f"{doc_id} §2", "paraphrase"),
    ]


def build_catering_doc(rng, doc_id, city, vendor):
    lead = rng.choice([3, 5, 7])
    minimum = rng.choice([10, 15, 20])
    sections = [
        ("Approved vendors",
         f"{vendor} is the approved on-site catering provider for events in {city}. External caterers "
         f"require prior written approval from the facilities manager."),
        ("Ordering and lead time",
         f"Catering orders must be placed at least {lead} business days before the event date, with a "
         f"minimum of {minimum} covers. Headcount changes are accepted up to 48 hours before service."),
        ("Dietary accommodation",
         f"{vendor} provides vegetarian, vegan and gluten-free options on request. Allergen information "
         f"is supplied per dish; severe allergy accommodation must be arranged directly with the vendor."),
    ]
    return _doc(doc_id, f"Catering services — {city}", sections), [
        (f"on-site and external catering options {city}", f"{doc_id} §1", "lexical"),
        (f"catering dietary accommodation in {city}", f"{doc_id} §3", "lexical"),
        (f"who supplies the food for {city} events", f"{doc_id} §1", "paraphrase"),
        (f"vegan and gluten free meals {city}", f"{doc_id} §3", "paraphrase"),
    ]


def build_travel_doc(rng, doc_id, dept, currency):
    per_diem = rng.choice([45, 60, 75, 90])
    days = rng.choice([14, 21, 30])
    sections = [
        ("Standard reimbursement",
         f"Employees in {dept} are reimbursed for economy class airfare, standard lodging, and meals up to "
         f"a per diem of {per_diem} {currency} per travel day, subject to itemised receipts."),
        ("Submission deadline",
         f"Expense claims must be submitted within {days} calendar days of the trip end date. Claims filed "
         f"after this window require line manager justification."),
        ("International travel",
         f"International travel requires senior director approval obtained before booking, and introduces a "
         f"mandatory foreign currency receipt verification step at claim review."),
        ("Non-reimbursable items",
         f"Personal entertainment, seat upgrades, traffic fines and companion travel costs are not "
         f"reimbursable under any circumstances."),
        ("Late booking exception",
         f"Where a booking is made after travel has commenced, reimbursement requires senior director "
         f"approval and a written explanation of the exception."),
    ]
    return _doc(doc_id, f"Travel and expense reimbursement — {dept}", sections), [
        (f"travel reimbursement rule for an employee trip in {dept}", f"{doc_id} §1", "lexical"),
        (f"international travel approval requirement for {dept}", f"{doc_id} §3", "lexical"),
        (f"post-travel booking exception for {dept}", f"{doc_id} §5", "lexical"),
        (f"how much money back per day on the road in {dept}", f"{doc_id} §1", "paraphrase"),
        (f"going abroad sign off needed {dept}", f"{doc_id} §3", "paraphrase"),
    ]


def build_support_doc(rng, doc_id, tier):
    hours = rng.choice([4, 8, 24])
    resolve = rng.choice([2, 3, 5])
    sections = [
        ("Response targets",
         f"{tier} support requests receive a first response within {hours} business hours of ticket creation."),
        ("Resolution targets",
         f"{tier} incidents are targeted for resolution within {resolve} business days. Progress updates "
         f"are provided at least once per business day until closure."),
        ("Escalation path",
         f"If a {tier} ticket exceeds its resolution target, it is escalated to the duty manager and, after "
         f"a further business day, to the regional support lead."),
    ]
    return _doc(doc_id, f"Support service levels — {tier}", sections), [
        (f"{tier} support response time", f"{doc_id} §1", "lexical"),
        (f"{tier} support escalation path", f"{doc_id} §3", "lexical"),
        (f"how quickly does someone reply to a {tier} ticket", f"{doc_id} §1", "paraphrase"),
        (f"who takes over a stuck {tier} ticket", f"{doc_id} §3", "paraphrase"),
    ]


def build_equipment_doc(rng, doc_id, city):
    port = rng.choice(["HDMI", "USB-C", "DisplayPort"])
    sections = [
        ("Connection options",
         f"Presentation systems in {city} meeting rooms accept {port} input. Adapters for legacy VGA are "
         f"available from reception on request."),
        ("Audio",
         f"Each room provides two wireless handheld microphones and one lapel microphone. Additional units "
         f"must be requested at least one business day in advance."),
        ("Fault reporting",
         f"Equipment faults are reported through the facilities helpdesk and are attended to within the same "
         f"business day where reported before 15:00 local time."),
    ]
    return _doc(doc_id, f"Meeting room equipment guide — {city}", sections), [
        (f"presentation connection port {city}", f"{doc_id} §1", "lexical"),
        (f"meeting room microphone availability in {city}", f"{doc_id} §2", "lexical"),
        (f"what cable do I need to plug in my laptop {city}", f"{doc_id} §1", "paraphrase"),
    ]


def build(seed: int, out_dir: Path) -> None:
    rng = random.Random(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qrels: list[dict] = []
    n = 0

    def emit(content: str, pairs, slug: str):
        nonlocal n
        doc_id = f"Doc_{n:02d}"
        (out_dir / f"{doc_id}_{slug}.md").write_text(content, encoding="utf-8")
        for query, ref, kind in pairs:
            qrels.append({"query": query, "relevant": [ref], "kind": kind})
        n += 1

    # The doc_id is assigned by emit(), so each builder is called with the id it will get.
    def next_id() -> str:
        return f"Doc_{n:02d}"

    for city in CITIES:
        for venue in rng.sample(VENUE_NAMES, 2):
            content, pairs = build_venue_doc(rng, next_id(), city, venue)
            emit(content, pairs, f"venue_{city.lower()}_{venue.split()[-1].lower()}")

    for city in CITIES:
        content, pairs = build_cancellation_doc(rng, next_id(), city)
        emit(content, pairs, f"cancellation_{city.lower()}")

    for city in rng.sample(CITIES, 6):
        content, pairs = build_catering_doc(rng, next_id(), city, rng.choice(VENDORS))
        emit(content, pairs, f"catering_{city.lower()}")

    for dept in DEPARTMENTS:
        content, pairs = build_travel_doc(rng, next_id(), dept, rng.choice(CURRENCIES))
        emit(content, pairs, f"travel_{dept.split()[0].lower()}")

    for tier in ["Standard", "Priority", "Critical"]:
        content, pairs = build_support_doc(rng, next_id(), tier)
        emit(content, pairs, f"support_{tier.lower()}")

    for city in rng.sample(CITIES, 4):
        content, pairs = build_equipment_doc(rng, next_id(), city)
        emit(content, pairs, f"equipment_{city.lower()}")

    with open(out_dir / "qrels.jsonl", "w", encoding="utf-8") as f:
        for row in qrels:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {n} documents and {len(qrels)} qrels to {out_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="fixtures/dev_corpus")
    args = parser.parse_args(argv)
    build(args.seed, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
