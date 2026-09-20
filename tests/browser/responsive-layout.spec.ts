import { expect, test } from "@playwright/test";
import snapshot from "../../assets/json/pubs.json";

const sizes = [
  [320, 256], [320, 480], [320, 568], [360, 400], [360, 640],
  [390, 340], [390, 844], [430, 400], [430, 932], [568, 320],
  [667, 375], [767, 900], [768, 1024], [769, 900], [820, 1180],
  [844, 390], [999, 900], [1000, 900], [1001, 900], [1024, 500],
  [1024, 501], [1024, 768], [1280, 360], [1280, 800], [1440, 900],
  [1920, 1080], [2560, 1440],
];

test("toolbar and empty states remain usable across the audited viewport matrix", async ({ page }) => {
  test.setTimeout(90_000);
  await page.goto("/?view=list");
  await expect(page.locator("#publication-search")).toBeEnabled();
  for (const [width, height] of sizes) {
    await page.setViewportSize({ width, height });
    for (const view of ["list", "map"]) {
      await page.locator(`#publication-view-${view}`).click();
      const controls = await page.locator("#graph-command-bar button, #publication-search").evaluateAll(elements =>
        elements.map(element => ({ id: element.id, ...element.getBoundingClientRect().toJSON() })));
      for (const box of controls) {
        expect(box.width, `${width}×${height} ${view} ${box.id}`).toBeGreaterThanOrEqual(44);
        expect(box.height).toBeGreaterThanOrEqual(44);
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.right).toBeLessThanOrEqual(width);
        expect(box.bottom).toBeLessThanOrEqual(height);
      }
      expect(controls.find(box => box.id === "publication-search")!.width).toBeGreaterThanOrEqual(150);
      for (let i = 0; i < controls.length; i++) {
        for (let j = i + 1; j < controls.length; j++) {
          const a = controls[i], b = controls[j];
          expect(a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top,
            `${width}×${height} ${a.id} overlaps ${b.id}`).toBe(true);
        }
      }
      await page.locator("#publication-search").fill("no-publication-matches-this-query");
      await expect(page.locator(".publication-results-help")).toBeHidden();
      const clear = page.locator("#publication-empty-clear");
      await clear.scrollIntoViewIfNeeded();
      await expect(clear).toBeInViewport({ ratio: 1 });
      await clear.click();
      await expect(page.locator("#publication-search")).toHaveValue("");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
  }
});

test("invalid paper warning reserves space and clears when a valid paper opens", async ({ page }) => {
  for (const width of [320, 1440]) {
    await page.setViewportSize({ width, height: 568 });
    await page.goto("/?view=list&paper=unknown-paper");
    await expect(page.locator("#graph-status")).toBeVisible();
    const status = await page.locator("#graph-status").boundingBox();
    const list = await page.locator("#publication-results").boundingBox();
    expect(list!.y).toBeGreaterThanOrEqual(status!.y + status!.height);
    await page.locator(".publication-result").first().click();
    await expect(page.locator("#graph-status")).toBeHidden();
    await expect(page.locator("#publication-detail")).toBeVisible();
  }
});

test("list and detail columns share the centered toolbar width", async ({ page }) => {
  await page.goto("/?view=list");
  await page.locator(".publication-result").first().click();
  for (const width of [1001, 1280, 1440, 1920, 2560]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(async () => {
      const bar = await page.locator("#graph-command-bar").boundingBox();
      const list = await page.locator("#publication-results").boundingBox();
      const detail = await page.locator("#publication-detail").boundingBox();
      return Math.max(Math.abs(list!.x - bar!.x),
        Math.abs(detail!.x + detail!.width - bar!.x - bar!.width));
    }).toBeLessThan(1);
  }
});

test("long details keep Close fixed and reveal copy feedback", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const paper = [...snapshot.records].sort((a, b) =>
    (b.bib_dict.abstract?.length || 0) - (a.bib_dict.abstract?.length || 0))[0];
  for (const viewport of [{ width: 390, height: 340 }, { width: 1440, height: 900 }]) {
    await page.setViewportSize(viewport);
    await page.goto(`/?view=list&paper=${encodeURIComponent(paper.author_pub_id)}`);
    await page.locator("#publication-abstract summary").click();
    const close = page.locator("#publication-detail-close");
    const before = await close.boundingBox();
    await page.locator("#publication-copy-citation").click();
    await expect(page.locator("#publication-copy-status")).toHaveText("Citation copied.");
    await expect(page.locator("#publication-copy-status")).toBeInViewport({ ratio: 1 });
    await expect(close).toBeInViewport({ ratio: 1 });
    expect((await close.boundingBox())!.y).toBe(before!.y);
  }
});

test("map labels and circles do not overlap, and keyboard focus scrolls into view", async ({ page }) => {
  for (const viewport of [{ width: 320, height: 568 }, { width: 1440, height: 900 }]) {
    await page.setViewportSize(viewport);
    await page.goto("/?view=map");
    await expect(page.locator(".cluster-label").first()).toBeVisible();
    const layout = await page.evaluate(() => {
      const labels = Array.from(document.querySelectorAll(".cluster-label"))
        .filter(label => getComputedStyle(label).display !== "none")
        .map(label => label.getBoundingClientRect());
      const labelCollisions = labels.flatMap((a, i) => labels.slice(i + 1).filter(b =>
        a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom));
      const circles = Array.from(document.querySelectorAll(".publication-node")).map(node => {
        const b = node.getBoundingClientRect();
        return { x: b.x + b.width / 2, y: b.y + b.height / 2, r: b.width / 2 };
      });
      const circleCollisions = circles.flatMap((a, i) => circles.slice(i + 1).filter(b =>
        Math.hypot(a.x - b.x, a.y - b.y) < a.r + b.r - 0.5));
      return { labels: labels.length, labelCollisions: labelCollisions.length, circleCollisions: circleCollisions.length };
    });
    expect(layout.labels).toBeGreaterThan(5);
    expect(layout.labelCollisions).toBe(0);
    expect(layout.circleCollisions).toBe(0);
    await page.locator(".publication-link").first().focus();
    await page.keyboard.press("End");
    await expect(page.locator(".publication-link").last()).toBeFocused();
    await expect(page.locator(".publication-link").last()).toBeInViewport({ ratio: 1 });
  }
});

test("tooltip is dismissed on resize and fits the new viewport when reopened", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/?view=map");
  await page.locator(".publication-link").first().focus();
  await expect(page.locator("#publication-tooltip")).toHaveAttribute("aria-hidden", "false");
  await page.setViewportSize({ width: 320, height: 568 });
  await expect(page.locator("#publication-tooltip")).toHaveAttribute("aria-hidden", "true");
  await page.locator("#publication-search").focus();
  await page.locator(".publication-link").first().focus();
  // Focusing an off-screen paper scrolls the canvas; refocus after that scroll.
  await expect(page.locator(".publication-link").first()).toBeInViewport();
  await page.locator("#publication-search").focus();
  await page.locator(".publication-link").first().focus();
  await expect(page.locator("#publication-tooltip")).toHaveAttribute("aria-hidden", "false");
  const box = await page.locator("#publication-tooltip").boundingBox();
  expect(box!.width).toBeGreaterThan(250);
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(320);
  expect(box!.y + box!.height).toBeLessThanOrEqual(568);
});

test("data failure disables unavailable views and offers Google Scholar directly", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 256 });
  await page.route("**/assets/json/pubs.json", route => route.abort());
  await page.goto("/?view=list");
  await expect(page.locator("#graph-status-message")).toContainText("temporarily unavailable");
  await expect(page.locator("#publication-view-list")).toBeDisabled();
  await expect(page.locator("#publication-view-map")).toBeDisabled();
  await expect(page.locator("#graph-status-scholar")).toHaveAttribute("href", /scholar.google.com\/citations\?user=.+/);
  await page.locator("#graph-status-scholar").scrollIntoViewIfNeeded();
  await expect(page.locator("#graph-status-scholar")).toBeInViewport({ ratio: 1 });
});
