import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

const snapshot = JSON.parse(readFileSync("assets/json/pubs.json", "utf8"));
const paper = snapshot.records[0];
const paperQuery = "Conceptual Design Generation";
const deepURL = `/?view=list&q=${encodeURIComponent(paperQuery)}&paper=${encodeURIComponent(paper.author_pub_id)}`;

async function listIDs(page: Page) {
  return page.locator("#publication-list li:not([hidden]) .publication-result").evaluateAll(
    buttons => buttons.map(button => button.getAttribute("data-publication-id")),
  );
}
async function noAxeViolations(page: Page) {
  const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  expect(scan.violations).toEqual([]);
}

test("list and map retain search, normalize punctuation, and rank titles first", async ({ page }) => {
  await page.goto("/");
  await page.locator("#exit").click();
  await expect(page.locator("#publication-view-map")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#publication-view-list").click();
  const search = page.locator("#publication-search");
  await search.fill("human-AI");
  const ids = await listIDs(page);
  expect(ids.length).toBeGreaterThan(3);
  for (const query of ["human ai", "human–AI", "human artificial intelligence"]) {
    await search.fill(query);
    expect(await listIDs(page)).toEqual(ids);
  }
  await page.locator("#publication-view-map").click();
  await expect(page.locator(".publication-link:not(.search-hidden)")).toHaveCount(ids.length);
  await expect(search).toHaveValue("human artificial intelligence");
  await page.locator("#publication-view-list").click();
  expect(await listIDs(page)).toEqual(ids);
  await search.fill(paperQuery);
  expect((await listIDs(page))[0]).toBe(paper.author_pub_id);
  await search.press("Enter");
  await expect(page.locator(".publication-result").filter({ hasText: paper.bib_dict.title })).toBeFocused();
  await noAxeViolations(page);
  await search.fill("nothing-matches-this-unusual-query");
  await expect(page.getByRole("heading", { name: "No publications found" })).toBeVisible();
  await page.locator("#publication-empty-clear").click();
  await expect(search).toHaveValue("");
  expect(await listIDs(page)).toHaveLength(snapshot.records.length);
});

for (const viewport of [{ width: 390, height: 844 }, { width: 320, height: 568 }, { width: 568, height: 320 }]) {
  test(`phone defaults to list with reachable details at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.locator("#exit").click();
    await expect(page.locator("#publication-view-list")).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("#publication-results")).toBeVisible();
    await expect(page.locator("#publication-graph")).toBeHidden();
    await noAxeViolations(page);
    await page.locator("#publication-search").fill(paperQuery);
    const result = page.locator(".publication-result").filter({ hasText: paper.bib_dict.title });
    await result.click();
    await expect(page.locator("#publication-detail-title")).toBeFocused();
    await expect(page.locator("#publication-results")).toBeHidden();
    const bounds = await page.locator("#publication-detail").boundingBox();
    const toolbar = await page.locator("#graph-command-bar").boundingBox();
    expect(bounds!.y).toBeGreaterThanOrEqual(toolbar!.y + toolbar!.height);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator("#publication-copy-link").scrollIntoViewIfNeeded();
    await expect(page.locator("#publication-copy-link")).toBeInViewport();
    await noAxeViolations(page);
    await page.keyboard.press("Escape");
    await expect(result).toBeFocused();
    await expect(page.locator("#publication-detail")).toBeHidden();
    await page.locator("#publication-view-map").click();
    await expect(page.locator("#publication-graph")).toBeVisible();
    await expect(page.locator("#publication-search")).toHaveValue(paperQuery);
  });
}

test("deep links survive refresh, dataset reordering, and back/forward", async ({ page }) => {
  await page.route("**/assets/json/pubs.json", route => route.fulfill({ json: {
    ...snapshot, records: [...snapshot.records].reverse(),
  } }));
  await page.goto(deepURL);
  await expect(page.locator("#publication-detail-title")).toHaveText(paper.bib_dict.title);
  await expect(page.locator("#publication-search")).toHaveValue(paperQuery);
  await page.reload();
  await expect(page.locator("#publication-detail-title")).toHaveText(paper.bib_dict.title);
  await page.locator("#publication-view-map").click();
  await expect(page).toHaveURL(/view=map/);
  await expect(page.locator("#publication-detail-title")).toHaveText(paper.bib_dict.title);
  await page.locator("#publication-detail-close").click();
  expect(new URL(page.url()).searchParams.has("paper")).toBe(false);
  await page.goBack();
  await expect(page.locator("#publication-detail-title")).toHaveText(paper.bib_dict.title);
  await page.goForward();
  await expect(page.locator("#publication-detail")).toBeHidden();
  await page.locator("#graph-close").click();
  await expect(page.locator("#profile")).toBeVisible();
  await page.goBack();
  await expect(page.locator("#publication-search")).toHaveValue(paperQuery);
});

test("direct paper links and clipboard actions contain the selected publication", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(deepURL);
  await expect(page.locator("#publication-resources a")).toHaveAttribute("href", paper.pub_url);
  await expect(page.locator("#publication-resources a")).toHaveAttribute("rel", /noopener/);
  await page.locator("#publication-copy-link").click();
  await expect(page.locator("#publication-copy-status")).toHaveText("Publication link copied.");
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(new URL(copied).searchParams.get("paper")).toBe(paper.author_pub_id);
  expect(new URL(copied).searchParams.get("q")).toBe(paperQuery);
  await page.locator("#publication-copy-citation").click();
  await expect(page.locator("#publication-copy-status")).toHaveText("Citation copied.");
  const citation = await page.evaluate(() => navigator.clipboard.readText());
  expect(citation).toContain(paper.bib_dict.title);
  expect(citation).toContain(paper.bib_dict.conference);
  expect(citation).toContain(paper.pub_url);
  await page.goto(copied);
  await expect(page.locator("#publication-detail-title")).toHaveText(paper.bib_dict.title);
});

test("copy failure has a selectable fallback and invalid paper URLs are recoverable", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", { value: { writeText: () => Promise.reject(new Error("denied")) } });
  });
  await page.goto(deepURL);
  await page.locator("#publication-copy-citation").click();
  await expect(page.locator("#publication-copy-fallback")).toBeVisible();
  await expect(page.locator("#publication-copy-fallback")).toHaveValue(new RegExp(paper.bib_dict.title));
  await expect(page.locator("#publication-copy-fallback")).toBeFocused();
  await page.goto("/?view=list&paper=unknown-paper");
  await expect(page.locator("#graph-status")).toContainText("not in the current collection");
  await page.locator("#publication-search").fill(paperQuery);
  await expect(page.locator("#graph-status")).toBeHidden();
  await page.locator(".publication-result").filter({ hasText: paper.bib_dict.title }).click();
  await expect(page.locator("#publication-detail-title")).toHaveText(paper.bib_dict.title);
});

test("unsafe resource metadata cannot become an executable link", async ({ page }) => {
  await page.route("**/assets/json/pubs.json", route => route.fulfill({ json: {
    ...snapshot, records: [{ ...paper, pub_url: "javascript:alert(1)", eprint_url: "data:text/html,<script>alert(1)</script>" }],
  } }));
  await page.goto(deepURL);
  await expect(page.locator("#publication-detail-title")).toHaveText(paper.bib_dict.title);
  await expect(page.locator("#publication-resources a")).toHaveCount(0);
  await expect(page.locator("#publication-detail-link")).toHaveAttribute("href", /^https:\/\/scholar.google.com/);
});

test("profile preview follows the responsive default and reveals the same list", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.locator("#publication-search")).toBeEnabled();
  await expect(page.locator("#publication-graph")).toBeVisible();
  await expect(page.locator("#publication-results")).toBeHidden();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("#exit")).toHaveAttribute("href", /view=list$/);
  await expect(page.locator("#publication-results")).toBeVisible();
  await expect(page.locator("#publication-graph")).toBeHidden();
  await expect(page.locator("#graph-container")).toHaveAttribute("inert", "");
  await expect(page.getByRole("button", { name: /Taylor Series Error Correction/ })).toHaveCount(0);
  const first = page.locator(".publication-result").first();
  const previewBounds = await first.boundingBox();
  const previewID = await first.getAttribute("data-publication-id");
  await page.locator("#exit").click();
  await expect(page.locator("#publication-view-list")).toHaveAttribute("aria-pressed", "true");
  expect(await first.getAttribute("data-publication-id")).toBe(previewID);
  const revealedBounds = await first.boundingBox();
  expect(Math.abs(revealedBounds!.y - previewBounds!.y)).toBeLessThan(1);
  await noAxeViolations(page);
});

test("chosen view persists behind the profile without background scrollbars after resizing", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/?view=list");
  await expect(page.locator("#publication-results")).toBeVisible();
  await page.locator("#graph-close").click();
  await expect(page.locator("#publication-results")).toBeVisible();
  await expect(page.locator("#publication-graph")).toBeHidden();
  for (const viewport of [{ width: 390, height: 844 }, { width: 1024, height: 500 }]) {
    await page.setViewportSize(viewport);
    await expect(page.locator("#publication-results")).toHaveCSS("overflow-y", "hidden");
    await expect(page.locator("#exit")).toHaveAttribute("href", /view=list$/);
  }
  await page.locator("#exit").click();
  await expect(page.locator("#publication-results")).toHaveCSS("overflow-y", "auto");
  await page.locator("#publication-view-map").click();
  await page.locator("#graph-close").click();
  for (const viewport of [{ width: 390, height: 340 }, { width: 820, height: 1180 }, { width: 1280, height: 360 }]) {
    await page.setViewportSize(viewport);
    await expect(page.locator("#publication-graph")).toBeVisible();
    await expect(page.locator("#publication-results")).toBeHidden();
    await expect(page.locator("#publication-map-viewport")).toHaveCSS("overflow", "hidden");
    await expect(page.locator("#exit")).toHaveAttribute("href", /view=map$/);
  }
  await page.locator("#exit").click();
  await expect(page.locator("#publication-map-viewport")).toHaveCSS("overflow", "auto");
  await expect(page.locator("#publication-view-map")).toHaveAttribute("aria-pressed", "true");
});
