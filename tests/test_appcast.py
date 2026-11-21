import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))
import make_appcast  # noqa: E402

SPARKLE = "{http://www.andymatuschak.org/xml-namespaces/sparkle}"


def test_notes_become_simple_html():
    html = make_appcast.notes_html(
        "One download for **every** Mac.\n\n## Changes\n\n- Faster `People`\n- See [the page](https://x.test/a?b=1&c=2)\n"
    )
    assert "<p>One download for <strong>every</strong> Mac.</p>" in html
    assert "<h3>Changes</h3>" in html
    assert "<ul><li>Faster <code>People</code></li><li>See <a href=\"https://x.test/a?b=1&amp;c=2\">the page</a></li></ul>" in html


def test_notes_never_carry_markup_from_the_text():
    assert "<script>" not in make_appcast.notes_html("Hello <script>alert(1)</script>")


def test_appcast_describes_one_signed_release():
    xml = make_appcast.appcast(
        version="0.1.3",
        url="https://github.com/o/r/releases/download/v0.1.3/Frame-Tidy-0.1.3.dmg",
        signature="c2lnbmF0dXJl",
        length=314000000,
        notes_html="<p>Hi</p>",
        minimum_system="12.3",
        published="Fri, 21 Nov 2026 10:00:00 +0000",
    )
    item = ET.fromstring(xml).find("channel/item")
    assert item.find(f"{SPARKLE}version").text == "0.1.3"
    assert item.find(f"{SPARKLE}shortVersionString").text == "0.1.3"
    assert item.find(f"{SPARKLE}minimumSystemVersion").text == "12.3"
    assert item.find("description").text == "<p>Hi</p>"
    enclosure = item.find("enclosure")
    assert enclosure.get("url").endswith("/v0.1.3/Frame-Tidy-0.1.3.dmg")
    assert enclosure.get(f"{SPARKLE}edSignature") == "c2lnbmF0dXJl"
    assert enclosure.get("length") == "314000000"


def test_signature_is_read_from_sign_update_output():
    out = 'sparkle:edSignature="QUJD+/=" length="12345"\n'
    assert make_appcast.parse_signature(out) == ("QUJD+/=", 12345)
    newer = 'sparkle:edSignature="QUJD" sparkle:length="6"\n'
    assert make_appcast.parse_signature(newer) == ("QUJD", 6)
