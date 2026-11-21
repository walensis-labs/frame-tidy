"""Write the appcast.xml that tells installed copies about a release.

    python3 packaging/make_appcast.py "dist/Frame-Tidy-0.1.3.dmg" 0.1.3 \\
        docs/release-notes/0.1.3.md dist/appcast.xml [--key-file KEY]

The appcast is attached to the GitHub release next to the dmg. Installed
copies read https://github.com/walensis-labs/frame-tidy/releases/latest/download/appcast.xml
(SUFeedURL in the app), so the newest release's appcast is always the one
they see, and it only needs to describe that release.

The dmg is signed with Sparkle's sign_update (build/sparkle/bin, from
packaging/fetch_sparkle.sh), using the private update key from the login
keychain, or from --key-file (GitHub Actions). Installed copies refuse an
update whose signature does not match the public key built into them.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

REPO = "walensis-labs/frame-tidy"
MINIMUM_SYSTEM = "12.3"
SIGN_UPDATE = Path(__file__).resolve().parent.parent / "build/sparkle/bin/sign_update"


def _inline(text: str) -> str:
    """Escape, then allow **bold**, `code` and [links](https://...)."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)

    def link(m: re.Match[str]) -> str:
        return f'<a href="{html.escape(html.unescape(m[2]))}">{m[1]}</a>'

    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", link, out)


def notes_html(markdown: str) -> str:
    """Release notes for Sparkle's window: paragraphs, ## headings and - lists only."""
    parts: list[str] = []
    items: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            parts.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()
        if items:
            parts.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in markdown.splitlines():
        line = line.strip()
        if not line:
            flush()
        elif line.startswith("#"):
            flush()
            parts.append(f"<h3>{_inline(line.lstrip('#').strip())}</h3>")
        elif line.startswith(("- ", "* ")):
            if para:
                flush()
            items.append(line[2:].strip())
        elif items:
            items[-1] += " " + line
        else:
            para.append(line)
    flush()
    return "".join(parts)


def parse_signature(output: str) -> tuple[str, int]:
    m = re.search(r'sparkle:edSignature="([^"]+)"\s+(?:sparkle:)?length="(\d+)"', output)
    if not m:
        raise SystemExit(f"sign_update gave no signature: {output.strip()}")
    return m[1], int(m[2])


def sign(dmg: Path, key_file: Path | None) -> tuple[str, int]:
    cmd = [str(SIGN_UPDATE), str(dmg)]
    if key_file:
        cmd[1:1] = ["--ed-key-file", str(key_file)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_signature(done.stdout)


def appcast(*, version: str, url: str, signature: str, length: int, notes_html: str,
            minimum_system: str = MINIMUM_SYSTEM, published: str | None = None) -> str:
    published = published or format_datetime(datetime.now(timezone.utc))
    notes = notes_html.replace("]]>", "]]&gt;")
    return f'''<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Frame Tidy</title>
    <link>https://walensis-labs.github.io/frame-tidy/</link>
    <item>
      <title>Version {escape(version)}</title>
      <pubDate>{escape(published)}</pubDate>
      <sparkle:version>{escape(version)}</sparkle:version>
      <sparkle:shortVersionString>{escape(version)}</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>{escape(minimum_system)}</sparkle:minimumSystemVersion>
      <description><![CDATA[{notes}]]></description>
      <enclosure url={quoteattr(url)} type="application/octet-stream" sparkle:edSignature={quoteattr(signature)} length="{length}"/>
    </item>
  </channel>
</rss>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dmg", type=Path)
    parser.add_argument("version")
    parser.add_argument("notes", type=Path, help="release notes (Markdown)")
    parser.add_argument("out", type=Path)
    parser.add_argument("--key-file", type=Path, help="private update key (default: login keychain)")
    parser.add_argument("--url", help="download address (default: the GitHub release asset)")
    args = parser.parse_args()
    url = args.url or f"https://github.com/{REPO}/releases/download/v{args.version}/{args.dmg.name}"
    signature, length = sign(args.dmg, args.key_file)
    args.out.write_text(appcast(version=args.version, url=url, signature=signature, length=length,
                                notes_html=notes_html(args.notes.read_text())))
    print(f"Wrote {args.out} for {args.version} ({length} bytes, signed).")


if __name__ == "__main__":
    main()
