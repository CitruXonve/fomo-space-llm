"""In-browser JavaScript IIFEs for LinkedIn feed scrolling and extraction."""

from __future__ import annotations

HIRING_JS_TEMPLATE = """
return (async () => {{
  const MIN_POSTS     = {min_posts};
  const MAX_SCROLLS   = {max_scrolls};
  const SCROLL_PX     = {scroll_px};
  const WAIT_MS       = {wait_ms};
  const PLATEAU_LIMIT = {plateau_limit};

  const HIRING_RE = /\\b(hiring|we'?re hiring|we\\s+are\\s+hiring|job opening|open role|apply(?:\\s+now| today)?|now hiring|looking for|seeking|join (?:our|the) team|open position|new role|job opportunit(?:y|ies)|recruiting|recruiter|vacancy|vacancies|career opportunity|job alert|accepting applications?|applications?\\s+(?:open|welcome|accepted?)|we(?:'re|\\s+are)\\s+looking|immediate opening|talent acquisition)\\b/i;
  const POST_URL_RE = /linkedin\\.com\\/(posts\\/[^?#"'\\s]+_activity|feed\\/update\\/)/;

  const feed = document.querySelector("div[data-testid=mainFeed]");
  if (!feed) return {{ error: "no feed" }};

  const seen = new Set();
  const results = [];
  let scrollStuck = 0;
  const ws = document.querySelector("#workspace") || document.documentElement;

  function postContainers() {{
    return [...feed.querySelectorAll(":scope > div[data-display-contents='true']")]
      .filter(c => c.querySelector("a[href]"));
  }}

  function extractPost(card) {{
    const hrefs = [...card.querySelectorAll("a[href]")].map(a => a.href);
    const companyUrls = [...new Set(
      hrefs.filter(h => {{
        try {{ return /^\\/company\\/[^/]+\\/?$/.test(new URL(h).pathname); }}
        catch {{ return false; }}
      }})
    )];
    const jobUrls   = [...new Set(hrefs.filter(h => /\\/jobs\\/view\\//.test(h)))];
    const authorUrl = hrefs.find(h => /linkedin\\.com\\/in\\//.test(h)) || null;
    const postUrl   = hrefs.find(h => POST_URL_RE.test(h)) || null;
    const text      = card.innerText || "";
    const isHiring  = jobUrls.length > 0 || HIRING_RE.test(text);
    if (!isHiring || !authorUrl) return null;
    const key = postUrl || authorUrl;
    if (seen.has(key)) return null;
    seen.add(key);
    return {{ company_urls: companyUrls, job_listing_urls: jobUrls, author_profile_url: authorUrl, post_url: postUrl }};
  }}

  for (let i = 0; i < MAX_SCROLLS; i++) {{
    for (const card of postContainers()) {{
      const entry = extractPost(card);
      if (entry) results.push(entry);
    }}
    if (results.length >= MIN_POSTS) break;
    const scrollBefore = ws.scrollTop;
    ws.scrollBy(0, SCROLL_PX);
    await new Promise(r => setTimeout(r, WAIT_MS));
    if (ws.scrollTop === scrollBefore) {{
      scrollStuck++;
      if (scrollStuck >= PLATEAU_LIMIT) break;
    }} else {{
      scrollStuck = 0;
    }}
  }}

  return results;
}})();
"""

# LinkedIn's current DOM uses ``div[data-lazy-mount-id]`` cards instead of legacy
# ``div[data-display-contents='true']`` children of mainFeed.
FEED_POSTS_JS_TEMPLATE = """
return (async () => {{
  const MIN_POSTS     = {min_posts};
  const MAX_SCROLLS   = {max_scrolls};
  const SCROLL_PX     = {scroll_px};
  const WAIT_MS       = {wait_ms};
  const PLATEAU_LIMIT = {plateau_limit};

  const POST_HREF_RE = /linkedin\\.com\\/(feed\\/update\\/|posts\\/)/;
  const URN_IN_STR_RE = /urn:li:(activity|share|ugcPost):\\d+/i;
  const URN_EXACT_RE = /^urn:li:(activity|share|ugcPost):\\d+$/i;

  const feedRoot = document.querySelector("div[data-testid=mainFeed]");
  const lazyCards = document.querySelectorAll("div[data-lazy-mount-id]");
  if (!feedRoot && !lazyCards.length) return {{ error: "no feed" }};
  const feed = feedRoot || document.querySelector("#workspace") || document.documentElement;

  const seen = new Set();
  const results = [];
  let scrollStuck = 0;
  const ws = document.querySelector("#workspace") || document.documentElement;

  function urnToFeedUrl(urn) {{
    const m = String(urn).match(URN_IN_STR_RE);
    if (!m) return null;
    return `https://www.linkedin.com/feed/update/${{m[0]}}/`;
  }}

  function findPostUrl(card) {{
    for (const a of card.querySelectorAll("a[href]")) {{
      const href = (a.href || "").split("?")[0];
      if (POST_HREF_RE.test(href)) return href;
    }}
    for (const el of card.querySelectorAll("*")) {{
      for (const attr of ["data-urn", "data-activity-urn"]) {{
        const raw = el.getAttribute(attr);
        if (!raw) continue;
        if (URN_EXACT_RE.test(raw)) {{
          const url = urnToFeedUrl(raw);
          if (url) return url;
        }}
        const embedded = raw.match(URN_IN_STR_RE);
        if (embedded) {{
          const url = urnToFeedUrl(embedded[0]);
          if (url) return url;
        }}
      }}
    }}
    return null;
  }}

  function postContainers() {{
    const legacy = feedRoot
      ? [...feedRoot.querySelectorAll(":scope > div[data-display-contents='true']")]
          .filter(c => c.querySelector("a[href], [data-urn], [data-activity-urn]"))
      : [];
    if (legacy.length) return legacy;
    const scope = feedRoot || document;
    return [...scope.querySelectorAll("div[data-lazy-mount-id]")]
      .filter(c => c.querySelector("a[href], [data-urn], [data-activity-urn]"));
  }}

  function extractPost(card) {{
    const postUrl = findPostUrl(card) || "";
    const hrefs = [...card.querySelectorAll("a[href]")].map(a => a.href);
    const authorUrl = hrefs.find(h =>
      /linkedin\\.com\\/in\\//.test(h) && !/\\/company\\//.test(h)
    ) || "";
    if (!postUrl && !authorUrl) return null;
    let text = (card.innerText || "").replace(/\\s+/g, " ").trim();
    if (text.length > 400) text = text.slice(0, 400) + "\\u2026";
    const timeEl = card.querySelector("time");
    const relTime = timeEl
      ? (timeEl.getAttribute("datetime") || timeEl.innerText || "").slice(0, 120)
      : "";
    const key = postUrl ? postUrl.split("?")[0] : `author:${{authorUrl}}`;
    if (seen.has(key)) return null;
    seen.add(key);
    return {{
      post_url: postUrl,
      author_profile_url: authorUrl,
      text_snippet: text,
      relative_time: relTime,
    }};
  }}

  for (let i = 0; i < MAX_SCROLLS; i++) {{
    for (const card of postContainers()) {{
      const entry = extractPost(card);
      if (entry) results.push(entry);
    }}
    if (results.length >= MIN_POSTS) break;
    const scrollBefore = ws.scrollTop;
    ws.scrollBy(0, SCROLL_PX);
    await new Promise(r => setTimeout(r, WAIT_MS));
    if (ws.scrollTop === scrollBefore) {{
      scrollStuck++;
      if (scrollStuck >= PLATEAU_LIMIT) break;
    }} else {{
      scrollStuck = 0;
    }}
  }}

  return results;
}})();
"""


def format_feed_posts_js(
    *,
    min_posts: int,
    max_scrolls: int,
    scroll_px: int,
    wait_ms: int,
    plateau_limit: int,
) -> str:
    return FEED_POSTS_JS_TEMPLATE.format(
        min_posts=min_posts,
        max_scrolls=max_scrolls,
        scroll_px=scroll_px,
        wait_ms=wait_ms,
        plateau_limit=plateau_limit,
    )


def format_hiring_js(
    *,
    min_posts: int,
    max_scrolls: int,
    scroll_px: int,
    wait_ms: int,
    plateau_limit: int,
) -> str:
    return HIRING_JS_TEMPLATE.format(
        min_posts=min_posts,
        max_scrolls=max_scrolls,
        scroll_px=scroll_px,
        wait_ms=wait_ms,
        plateau_limit=plateau_limit,
    )
