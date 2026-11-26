import re

from dhs import site


def test_page_builds_with_every_placeholder_filled():
    html = site.build_html()
    assert not re.search(r"\$(?!\{)[a-z_]", html)  # JS template literals use ${...}
    assert "function hedgePnl" in html
    assert 'id="hedging-simulator"' in html and 'id="method-and-caveats"' in html
    assert "'" not in re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", "", html, flags=re.S)
