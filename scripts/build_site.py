"""Build site/index.html, the hedging simulator, from code alone (no data needed)."""

from pathlib import Path

from dhs.site import build_html


def main() -> None:
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
