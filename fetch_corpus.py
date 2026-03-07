"""
Download and process the Enron Email Dataset (CMU source).

Corpus: Enron Email Dataset — email threads, forwarding/quoting, identity
resolution challenges. As suggested in the assignment.

Source: CMU Enron dataset (https://www.cs.cmu.edu/~enron/)
Download: enron_mail_20150507.tar.gz (~1.7 GB)

The script:
1. Downloads the tar.gz from CMU (with resume support)
2. Extracts only selected user mailboxes (saves disk space)
3. Parses raw email files into structured messages
4. Groups messages into conversation threads
5. Selects 50-75 threads with 3+ messages each
6. Saves as data/raw_corpus.json in the pipeline's expected format
"""

import email
import email.policy
import hashlib
import json
import os
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "raw_corpus.json")
ENRON_TAR = os.path.join(DATA_DIR, "enron_mail_20150507.tar.gz")
ENRON_EXTRACT_DIR = os.path.join(DATA_DIR, "enron_maildir")

CMU_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

# Users with rich email threads — known active Enron correspondents
# These cover: executive decisions, legal discussion, trading, research, policy
TARGET_USERS = [
    "kaminski-v",    # Vince Kaminski — VP Research, technical + business
    "dasovich-j",    # Jeff Dasovich — Government Affairs, policy debates
    "mann-k",        # Kay Mann — Legal, discussions & decisions
    "shackleton-s",  # Sara Shackleton — Trading, commercial threads
    "lay-k",         # Kenneth Lay — CEO, executive decisions
    "germany-c",     # Chris Germany — Trading, operational discussions
    "farmer-d",      # Daren Farmer — Logistics, scheduling threads
]

# Folders within each user's mailbox to extract
TARGET_FOLDERS = ["sent_items", "sent", "inbox", "all_documents", "_sent_mail"]

MIN_MESSAGES_PER_THREAD = 5
MAX_MESSAGES_PER_THREAD = 15
TARGET_THREAD_COUNT = 15


def download_enron():
    """Download the Enron tar.gz from CMU with resume support (pure Python)."""
    if os.path.exists(ENRON_TAR):
        size_mb = os.path.getsize(ENRON_TAR) / (1024 * 1024)
        if size_mb > 1500:
            print(f"  Enron tar.gz already downloaded ({size_mb:.0f} MB). Skipping.")
            return True
        print(f"  Partial download found ({size_mb:.0f} MB). Resuming...")

    print(f"  Downloading Enron Email Dataset from CMU...")
    print(f"  URL: {CMU_URL}")
    print(f"  This is ~1.7 GB. It will resume if interrupted.")
    print()

    try:
        # Resume support via Range header
        headers = {}
        mode = "wb"
        downloaded = 0
        if os.path.exists(ENRON_TAR):
            downloaded = os.path.getsize(ENRON_TAR)
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"

        resp = requests.get(CMU_URL, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0)) + downloaded
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(ENRON_TAR, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\r  Progress: {downloaded/(1024*1024):.0f}/{total/(1024*1024):.0f} MB ({pct:.1f}%)", end="", flush=True)

        print(f"\n  Download complete.")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print(f"  You can also manually download from:")
        print(f"    {CMU_URL}")
        print(f"  And place it at: {ENRON_TAR}")
        return False


def extract_mailboxes():
    """Extract only the target users' mailboxes from the tar.gz (pure Python)."""
    if os.path.exists(ENRON_EXTRACT_DIR) and any(
        os.path.isdir(os.path.join(ENRON_EXTRACT_DIR, u))
        for u in TARGET_USERS
    ):
        count = sum(1 for u in TARGET_USERS
                    if os.path.isdir(os.path.join(ENRON_EXTRACT_DIR, u)))
        print(f"  Mailboxes already extracted ({count} users). Skipping.")
        return True

    os.makedirs(ENRON_EXTRACT_DIR, exist_ok=True)

    # Prefixes to match inside the tar: maildir/<user>/
    target_prefixes = tuple(f"maildir/{u}/" for u in TARGET_USERS)

    print(f"  Extracting {len(TARGET_USERS)} user mailboxes...")
    print(f"  (This reads through the full archive but only extracts target users)")

    try:
        with tarfile.open(ENRON_TAR, "r:gz") as tar:
            for member in tar:
                if not member.name.startswith(target_prefixes):
                    continue
                # Strip "maildir/" prefix (equivalent to --strip-components=1)
                member.name = member.name[len("maildir/"):]
                # Security: skip absolute paths or path traversal
                if member.name.startswith(("/", "..")) or ".." in member.name:
                    continue
                tar.extract(member, path=ENRON_EXTRACT_DIR, filter="data")
    except Exception as e:
        print(f"  Extraction error: {e}")
        # Try without filter= for older Python (<3.12)
        try:
            with tarfile.open(ENRON_TAR, "r:gz") as tar:
                for member in tar:
                    if not member.name.startswith(target_prefixes):
                        continue
                    member.name = member.name[len("maildir/"):]
                    if member.name.startswith(("/", "..")) or ".." in member.name:
                        continue
                    tar.extract(member, path=ENRON_EXTRACT_DIR)
        except Exception as e2:
            print(f"  Extraction failed: {e2}")
            return False

    extracted = [u for u in TARGET_USERS
                 if os.path.isdir(os.path.join(ENRON_EXTRACT_DIR, u))]
    print(f"  Extracted {len(extracted)} user mailboxes: {', '.join(extracted)}")
    return len(extracted) > 0


def parse_email_file(filepath: str) -> dict:
    """Parse a single raw email file into a structured dict."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            msg = email.message_from_file(f, policy=email.policy.default)
    except Exception:
        return None

    # Extract headers
    msg_id = msg.get("Message-ID", "")
    sender = msg.get("From", "unknown")
    to_raw = msg.get("To", "")
    cc_raw = msg.get("Cc", "")
    date_str = msg.get("Date", "")
    subject = msg.get("Subject", "(no subject)")
    in_reply_to = msg.get("In-Reply-To", "")
    references = msg.get("References", "")

    # Parse body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="replace")

    if not body.strip():
        return None

    # Parse date to ISO format
    timestamp = ""
    if date_str:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            timestamp = dt.isoformat()
        except Exception:
            timestamp = date_str

    # Parse recipient lists
    def parse_addrs(raw):
        if not raw:
            return []
        # Handle comma-separated email addresses
        addrs = [a.strip() for a in str(raw).split(",")]
        return [a for a in addrs if a and "@" in a]

    return {
        "message_id": msg_id.strip("<> ") if msg_id else hashlib.md5(
            f"{sender}{date_str}{subject}".encode()
        ).hexdigest()[:12],
        "sender": str(sender).strip(),
        "to": parse_addrs(to_raw),
        "cc": parse_addrs(cc_raw),
        "subject": str(subject).strip(),
        "timestamp": timestamp,
        "body": body.strip(),
        "in_reply_to": in_reply_to.strip("<> ") if in_reply_to else "",
        "references": references.strip() if references else "",
        "filepath": filepath,
    }


def collect_all_emails() -> list[dict]:
    """Walk extracted mailboxes and parse all email files."""
    all_emails = []
    seen_ids = set()

    for user in TARGET_USERS:
        user_dir = os.path.join(ENRON_EXTRACT_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        count = 0
        for folder in TARGET_FOLDERS:
            folder_path = os.path.join(user_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                if not os.path.isfile(fpath):
                    continue

                parsed = parse_email_file(fpath)
                if parsed and parsed["message_id"] not in seen_ids:
                    all_emails.append(parsed)
                    seen_ids.add(parsed["message_id"])
                    count += 1

        print(f"  {user}: {count} emails parsed")

    print(f"  Total unique emails: {len(all_emails)}")
    return all_emails


def normalize_subject(subject: str) -> str:
    """Normalize subject line for thread grouping."""
    s = subject.strip()
    # Remove Re:/Fwd:/FW:/RE: prefixes (possibly repeated)
    s = re.sub(r"^(Re|Fwd|FW|RE|Fw):\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(Re|Fwd|FW|RE|Fw):\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(Re|Fwd|FW|RE|Fw):\s*", "", s, flags=re.IGNORECASE)
    s = s.strip()
    return s.lower()


def group_into_threads(emails: list[dict]) -> list[list[dict]]:
    """Group emails into conversation threads by subject + reply chains."""
    # Build reply map: message_id -> email
    by_id = {}
    for em in emails:
        by_id[em["message_id"]] = em

    # Phase 1: Group by normalized subject
    subject_groups = defaultdict(list)
    for em in emails:
        norm_subj = normalize_subject(em["subject"])
        if norm_subj:
            subject_groups[norm_subj].append(em)

    # Phase 2: Sort each group chronologically
    threads = []
    for subj, group in subject_groups.items():
        group.sort(key=lambda e: e.get("timestamp", ""))
        threads.append(group)

    return threads


def build_corpus(threads: list[list[dict]]) -> list[dict]:
    """
    Convert email threads into the pipeline's expected corpus format.
    Each thread becomes an "issue" with "comments".
    """
    # Filter: keep threads with MIN..MAX messages (avoids mega-threads)
    good_threads = [
        t for t in threads
        if MIN_MESSAGES_PER_THREAD <= len(t) <= MAX_MESSAGES_PER_THREAD
    ]

    # Sort by thread length (most emails first), take TARGET_THREAD_COUNT
    good_threads.sort(key=lambda t: len(t), reverse=True)
    good_threads = good_threads[:TARGET_THREAD_COUNT]

    corpus = []
    for idx, thread in enumerate(good_threads):
        thread_id = idx + 1
        subject = thread[0]["subject"]

        # Cap thread length
        thread = thread[:MAX_MESSAGES_PER_THREAD]

        # Build comments from each email in thread
        comments = []
        for i, em in enumerate(thread):
            comments.append({
                "comment_id": em["message_id"][:32],
                "author": em["sender"],
                "timestamp": em["timestamp"],
                "text": em["body"],
                "is_issue_body": (i == 0),
            })

        # Determine thread time range
        timestamps = [em["timestamp"] for em in thread if em["timestamp"]]
        created_at = min(timestamps) if timestamps else ""
        closed_at = max(timestamps) if timestamps else ""

        # Collect unique participants
        participants = set()
        for em in thread:
            participants.add(em["sender"])
            participants.update(em.get("to", []))

        corpus.append({
            "issue_id": thread_id,
            "title": subject,
            "state": "closed",
            "author": thread[0]["sender"],
            "created_at": created_at,
            "closed_at": closed_at,
            "labels": [],
            "url": f"enron://thread/{thread_id}",
            "comments": comments,
        })

    return corpus


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if corpus already exists
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        print(f"Corpus already exists: {len(existing)} threads. To re-fetch, delete {OUTPUT_PATH}")
        return

    print("=" * 60)
    print("ENRON EMAIL DATASET — CORPUS DOWNLOAD & PROCESSING")
    print("=" * 60)
    print(f"Source: CMU Enron dataset")
    print(f"URL: {CMU_URL}")
    print()

    # Step 1: Download
    print("[Step 1/4] Downloading Enron Email Dataset...")
    if not download_enron():
        sys.exit(1)

    # Step 2: Extract mailboxes
    print(f"\n[Step 2/4] Extracting {len(TARGET_USERS)} user mailboxes...")
    if not extract_mailboxes():
        print("  Failed to extract mailboxes.")
        sys.exit(1)

    # Step 3: Parse emails
    print(f"\n[Step 3/4] Parsing emails...")
    all_emails = collect_all_emails()
    if len(all_emails) < 50:
        print(f"  Only {len(all_emails)} emails found. Need more data.")
        sys.exit(1)

    # Step 4: Thread and build corpus
    print(f"\n[Step 4/4] Grouping into threads and building corpus...")
    threads = group_into_threads(all_emails)
    print(f"  Found {len(threads)} threads total")
    print(f"  Threads with {MIN_MESSAGES_PER_THREAD}+ messages: "
          f"{sum(1 for t in threads if len(t) >= MIN_MESSAGES_PER_THREAD)}")

    corpus = build_corpus(threads)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    total_messages = sum(len(t["comments"]) for t in corpus)
    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(corpus)} email threads with {total_messages} "
          f"total messages to {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
