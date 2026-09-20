import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const SITE_ORIGIN = "https://cmccomb.com";
const TEST_ORIGIN = "http://127.0.0.1:4173";
const SCHOLAR_URL_PATTERN = /^https:\/\/scholar\.google\.com\/citations\?/;
const HEADSHOT_URL = `${SITE_ORIGIN}/assets/images/headshot_optimized_square.jpg`;
const CMU_PROFILE_URL =
  "https://meche.engineering.cmu.edu/directory/bios/mccomb-christopher.html";
const CV_PATH = "/assets/files/Christopher-McComb-CV.pdf";

async function expectNoAccessibilityViolations(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();

  expect(
    results.violations,
    results.violations
      .map((violation) => `${violation.id}: ${violation.help}`)
      .join("\n"),
  ).toEqual([]);
}

test.describe("homepage", () => {
  test("has semantic metadata and no initial accessibility violations", async ({
    page,
  }) => {
    await page.goto("/");

    await expect(page.getByRole("main")).toHaveCount(1);
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: "Chris McComb, Ph.D.",
      }),
    ).toBeVisible();

    const description = page.locator('meta[name="description"]');
    await expect(description).toHaveAttribute("content", /professional profile/i);
    await expect(description).not.toHaveAttribute("content", /\.\.$/);
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
      "href",
      `${SITE_ORIGIN}/`,
    );

    await expect(page.locator('meta[property="og:type"]')).toHaveAttribute(
      "content",
      "profile",
    );
    await expect(page.locator('meta[property="og:url"]')).toHaveAttribute(
      "content",
      `${SITE_ORIGIN}/`,
    );
    await expect(page.locator('meta[property="og:image"]')).toHaveAttribute(
      "content",
      HEADSHOT_URL,
    );
    await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute(
      "content",
      "summary",
    );
    await expect(page.locator('meta[name="twitter:image"]')).toHaveAttribute(
      "content",
      HEADSHOT_URL,
    );

    const structuredData = page.locator('script[type="application/ld+json"]');
    await expect(structuredData).toHaveCount(1);
    const person = JSON.parse((await structuredData.textContent()) ?? "{}");
    expect(person).toMatchObject({
      "@context": "https://schema.org",
      "@type": "Person",
      name: "Chris McComb",
      url: `${SITE_ORIGIN}/`,
      image: HEADSHOT_URL,
      email: "mailto:ccm@cmu.edu",
      jobTitle: "Professor, Mechanical Engineering",
      worksFor: {
        "@type": "CollegeOrUniversity",
        name: "Carnegie Mellon University",
        url: "https://www.cmu.edu/",
      },
      affiliation: {
        "@type": "CollegeOrUniversity",
        name: "Carnegie Mellon University",
        url: "https://www.cmu.edu/",
      },
    });
    expect(person.sameAs).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/^https:\/\/github\.com\//),
        expect.stringMatching(/^https:\/\/www\.linkedin\.com\//),
        expect.stringMatching(SCHOLAR_URL_PATTERN),
        "https://x.com/ccmccomb",
        CMU_PROFILE_URL,
      ]),
    );

    const graph = page.locator("#graph-container");
    await expect(graph).toHaveAttribute("aria-hidden", "true");
    await expect(graph).toHaveAttribute("inert", "");

    const contactLinks = page
      .getByRole("group", { name: "Contact and profile links" })
      .getByRole("link");
    await expect(contactLinks).toHaveCount(5);

    const linkedin = page.getByRole("link", { name: "LinkedIn" });
    await expect(linkedin.locator("svg")).toHaveClass(/linkedin-icon/);

    const cv = page.getByRole("link", { name: "Curriculum vitae (PDF)" });
    await expect(cv).toHaveAttribute("href", CV_PATH);
    await expect(cv).toHaveAttribute("target", "_blank");
    await expect(cv).toHaveAttribute("rel", /noopener/);

    const contactBounds = await contactLinks.evaluateAll((links) =>
      links.map((link) => link.getBoundingClientRect().toJSON()),
    );
    expect(contactBounds.every((bounds) => bounds.height >= 44)).toBe(true);
    const contactWidths = contactBounds.map((bounds) => bounds.width);
    expect(Math.max(...contactWidths) - Math.min(...contactWidths)).toBeLessThan(1);

    const cvResponse = await page.request.get(CV_PATH);
    expect(cvResponse.ok()).toBe(true);
    expect(cvResponse.headers()["content-type"]).toMatch(/application\/pdf/i);
    expect((await cvResponse.body()).subarray(0, 5).toString()).toBe("%PDF-");

    await expectNoAccessibilityViolations(page);
  });

  test("keeps controls reachable without horizontal overflow in short landscape", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 667, height: 320 });
    await page.goto("/");

    const publicationControl = page.locator("#exit");
    await expect(publicationControl).toBeVisible();
    await publicationControl.scrollIntoViewIfNeeded();
    await publicationControl.focus();
    await expect(publicationControl).toBeFocused();

    const controlBox = await publicationControl.boundingBox();
    expect(controlBox).not.toBeNull();
    expect(controlBox?.y).toBeGreaterThanOrEqual(0);
    expect((controlBox?.y ?? 0) + (controlBox?.height ?? 0)).toBeLessThanOrEqual(
      320,
    );

    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  });

  test("activates the graph, supports keyboard navigation, and restores focus", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/");

    const graph = page.locator("#graph-container");
    const explore = page.locator("#exit");
    const close = page.locator("#graph-close");

    await expect(explore).toHaveAttribute("aria-expanded", "false");
    await expect(page.locator(".publication-link")).not.toHaveCount(0);
    const nodePositionsBeforeReveal = await page
      .locator(".publication-node")
      .evaluateAll((nodes) => nodes.map((node) => ({
        x: node.getAttribute("x"),
        y: node.getAttribute("y"),
      })));

    await explore.click();
    await expect(graph).toHaveAttribute("aria-hidden", "false");
    await expect(graph).not.toHaveAttribute("inert", "");
    await expect(explore).toHaveAttribute("aria-expanded", "true");
    await expect(close).toBeVisible();
    await expect(close).toHaveAttribute("aria-label", "Back to profile");
    await expect(page.locator("#graph-command-bar")).toBeVisible();
    const commandBarBounds = await page.locator("#graph-command-bar").boundingBox();
    const closeBounds = await close.boundingBox();
    const closeInsets = {
      left: (closeBounds?.x ?? 0) - (commandBarBounds?.x ?? 0),
      top: (closeBounds?.y ?? 0) - (commandBarBounds?.y ?? 0),
      bottom:
        (commandBarBounds?.y ?? 0)
        + (commandBarBounds?.height ?? 0)
        - (closeBounds?.y ?? 0)
        - (closeBounds?.height ?? 0),
    };
    expect(closeBounds?.width).toBe(44);
    expect(closeBounds?.height).toBe(44);
    expect(Math.abs(closeInsets.left - closeInsets.bottom)).toBeLessThan(1);
    const searchBounds = await page.locator("#publication-search").boundingBox();
    expect(
      Math.abs(
        ((searchBounds?.x ?? 0) - (closeBounds?.x ?? 0) - (closeBounds?.width ?? 0))
        - closeInsets.left,
      ),
    ).toBeLessThan(2);
    await expect(page.locator("#publication-search")).toBeFocused();
    const nodePositionsAfterReveal = await page
      .locator(".publication-node")
      .evaluateAll((nodes) => nodes.map((node) => ({
        x: node.getAttribute("x"),
        y: node.getAttribute("y"),
      })));
    expect(nodePositionsAfterReveal).toEqual(nodePositionsBeforeReveal);
    await expect(page.locator("#publication-search-help")).toHaveCount(0);
    await expect(page.locator(".colorbar-legend svg")).toBeVisible();
    await expect(page.locator(".size-legend svg")).toBeVisible();
    await expect(page.locator(".size-legend")).toHaveAttribute(
      "aria-label",
      /citation counts from \d+ to \d+/i,
    );
    const citationLegendRadii = await page
      .locator(".citation-size-key circle")
      .evaluateAll((circles) => circles.map((circle) => Number(circle.getAttribute("r"))));
    expect(citationLegendRadii).toHaveLength(3);
    expect(citationLegendRadii[0]).toBeLessThan(citationLegendRadii[1]);
    expect(citationLegendRadii[1]).toBeLessThan(citationLegendRadii[2]);
    expect(citationLegendRadii[2]).toBeLessThanOrEqual(20);
    const largestCitationKey = page.locator(".citation-size-key").last();
    const largestCitationCircleBounds = await largestCitationKey
      .locator("circle")
      .boundingBox();
    const largestCitationLabelBounds = await largestCitationKey
      .locator("text")
      .boundingBox();
    expect(
      (largestCitationLabelBounds?.y ?? 0)
      - (largestCitationCircleBounds?.y ?? 0)
      - (largestCitationCircleBounds?.height ?? 0),
    ).toBeGreaterThanOrEqual(2);
    await expectNoAccessibilityViolations(page);

    const firstPublication = page.locator(".publication-link").first();
    const secondPublication = page.locator(".publication-link").nth(1);
    await expect(firstPublication).toHaveAttribute("tabindex", "0");
    await expect(secondPublication).toHaveAttribute("tabindex", "-1");
    await firstPublication.focus();
    await expect(firstPublication).toBeFocused();

    await firstPublication.press("ArrowRight");
    await expect(secondPublication).toBeFocused();
    await expect(secondPublication).toHaveAttribute("role", "button");
    await expect(secondPublication).not.toHaveAttribute("href", /.+/);
    await page.context().route("https://scholar.google.com/**", async (route) => {
      await route.fulfill({
        contentType: "text/html",
        body: "<title>Google Scholar</title>",
      });
    });
    await secondPublication.press("Enter");
    await expect(page).toHaveURL(/\?view=/);
    await expect(secondPublication).toHaveAttribute("aria-expanded", "true");

    const detail = page.locator("#publication-detail");
    const detailLink = page.locator("#publication-detail-link");
    await expect(detail).toBeVisible();
    await expect(page.locator("#publication-detail-title")).not.toBeEmpty();
    await expect(detailLink).toHaveAttribute("href", SCHOLAR_URL_PATTERN);
    await expect(detailLink).toHaveAttribute("target", "_blank");
    await expect(detailLink).toHaveAttribute("rel", /noopener/);
    await expectNoAccessibilityViolations(page);

    const popupPromise = page.waitForEvent("popup");
    await detailLink.click();
    const popup = await popupPromise;
    await expect(popup).toHaveURL(SCHOLAR_URL_PATTERN);
    await popup.close();

    await page.locator("#publication-detail-close").click();
    await expect(detail).toBeHidden();
    await expect(secondPublication).toBeFocused();

    await secondPublication.press(" ");
    await expect(detail).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(detail).toBeHidden();
    await expect(secondPublication).toBeFocused();

    await close.click();
    await expect(graph).toHaveAttribute("aria-hidden", "true");
    await expect(graph).toHaveAttribute("inert", "");
    await expect(explore).toHaveAttribute("aria-expanded", "false");
    await expect(explore).toBeFocused();

    await explore.click();
    await expect(page.locator("#publication-search")).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(graph).toHaveAttribute("aria-hidden", "true");
    await expect(explore).toBeFocused();
  });

  test("searches publications and uses Escape to unwind map state", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await page.locator("#exit").click();

    const graph = page.locator("#graph-container");
    const search = page.locator("#publication-search");
    const status = page.locator("#publication-search-status");
    const allPublications = page.locator(".publication-link");
    const total = await allPublications.count();
    expect(total).toBeGreaterThan(1);
    await expect(status).toHaveText(`${total} publications`);

    await search.fill("no-publication-can-match-this-phrase");
    await expect(status).toHaveText(`0 of ${total} publications`);
    await expect(page.locator('.publication-link[tabindex="0"]')).toHaveCount(0);
    await expect(page.locator(".publication-link.search-hidden")).toHaveCount(total);
    await page.locator("#publication-search-clear").click();

    await search.fill("large language");
    const matches = page.locator(".publication-link:not(.search-hidden)");
    const nonmatches = page.locator(".publication-link.search-hidden");
    const matchCount = await matches.count();
    expect(matchCount).toBeGreaterThan(1);
    expect(matchCount).toBeLessThan(total);
    await expect(status).toHaveText(`${matchCount} of ${total} publications`);
    await expect(nonmatches.first()).toHaveAttribute("aria-hidden", "true");
    await expect(nonmatches.first()).toHaveAttribute("tabindex", "-1");

    await search.press("Enter");
    await expect(page.locator('.publication-link[tabindex="0"]')).toBeFocused();
    await page.keyboard.press("Home");
    await expect(matches.first()).toBeFocused();
    await matches.first().press("ArrowRight");
    await expect(matches.nth(1)).toBeFocused();
    await matches.nth(1).press("End");
    await expect(matches.last()).toBeFocused();
    await matches.last().press("ArrowRight");
    await expect(matches.first()).toBeFocused();

    await matches.first().click();
    await expect(page).toHaveURL(/\?view=/);
    await expect(page.locator("#publication-detail")).toBeVisible();
    await expect(page.locator('.publication-link[aria-expanded="true"]')).toHaveCount(1);

    await page.keyboard.press("Escape");
    await expect(page.locator("#publication-detail")).toBeHidden();
    await expect(search).toHaveValue("large language");
    await expect(graph).toHaveAttribute("aria-hidden", "false");

    await page.keyboard.press("Escape");
    await expect(search).toHaveValue("");
    await expect(status).toHaveText(`${total} publications`);
    await expect(nonmatches).toHaveCount(0);
    await expect(graph).toHaveAttribute("aria-hidden", "false");

    await page.keyboard.press("Escape");
    await expect(graph).toHaveAttribute("aria-hidden", "true");
    await expect(page.locator("#exit")).toBeFocused();

    await page.locator("#exit").click();
    await expect(search).toHaveValue("");
    await expect(page.locator("#publication-detail")).toBeHidden();
    await expect(allPublications.first()).toHaveAttribute("tabindex", "0");

    await search.fill("   ");
    await expect(page.locator("#publication-search-clear")).toBeEnabled();
    await search.press("Escape");
    await expect(search).toHaveValue("");
    await expect(graph).toHaveAttribute("aria-hidden", "false");
    await expectNoAccessibilityViolations(page);
  });

  test("opens and closes the responsive publication map on mobile", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    const graph = page.locator("#graph-container");
    const explore = page.locator("#exit");
    const close = page.locator("#graph-close");
    await expect(explore).toHaveAttribute("href", /\?view=list$/);
    await expect(explore).toHaveText("Explore publications");
    const exploreBounds = await explore.boundingBox();
    expect(exploreBounds?.height).toBeLessThanOrEqual(40);

    await explore.click();
    await expect(page.locator("#publication-results")).toBeVisible();
    await page.locator("#publication-view-map").click();
    await page.locator("#publication-search").focus();
    await expect(page).toHaveURL(/\?view=/);
    await expect(graph).toHaveAttribute("aria-hidden", "false");
    await expect(close).toBeVisible();
    await expect(page.locator("#publication-search")).toBeFocused();
    await expect(page.locator(".publication-link").first()).toHaveAttribute(
      "tabindex",
      "0",
    );

    const mobileLayout = await page.evaluate(() => {
      const viewportWidth = document.documentElement.clientWidth;
      const viewportHeight = document.documentElement.clientHeight;
      const graphBounds = document
        .getElementById("graph-container")
        ?.getBoundingClientRect();
      const closeBounds = document
        .getElementById("graph-close")
        ?.getBoundingClientRect();
      const legendBounds = document
        .querySelector(".colorbar-legend")
        ?.getBoundingClientRect();
      const pointBounds = Array.from(
        document.querySelectorAll(".publication-node"),
      ).map((point) => point.getBoundingClientRect());
      const canvas = document.getElementById("publication-graph")!.getBoundingClientRect();
      const labelsAreInCanvas = Array.from(
        document.querySelectorAll(".cluster-label"),
      ).filter(label => getComputedStyle(label).display !== "none").every((label) => {
        const bounds = label.getBoundingClientRect();
        return (
          bounds.left >= canvas.left
          && bounds.top >= canvas.top
          && bounds.right <= canvas.right
          && bounds.bottom <= canvas.bottom
        );
      });

      return {
        graphHeight: graphBounds?.height,
        graphWidth: graphBounds?.width,
        closeIsInViewport: Boolean(
          closeBounds
          && closeBounds.left >= 0
          && closeBounds.top >= 0
          && closeBounds.right <= viewportWidth
          && closeBounds.bottom <= viewportHeight
        ),
        hasHorizontalOverflow:
          document.documentElement.scrollWidth > viewportWidth,
        pointsOverlapLegend: Boolean(
          legendBounds
          && pointBounds.some((bounds) => (
            bounds.left < legendBounds.right
            && bounds.right > legendBounds.left
            && bounds.top < legendBounds.bottom
            && bounds.bottom > legendBounds.top
          ))
        ),
        smallestPointTarget: Math.min(
          ...pointBounds.map((bounds) => Math.min(bounds.width, bounds.height)),
        ),
        pointsAreInCanvas: pointBounds.every((bounds) => (
          bounds.left >= canvas.left
          && bounds.top >= canvas.top
          && bounds.right <= canvas.right
          && bounds.bottom <= canvas.bottom
        )),
        labelsAreInCanvas,
        viewportHeight,
        viewportWidth,
      };
    });

    expect(mobileLayout.graphWidth).toBe(mobileLayout.viewportWidth);
    expect(mobileLayout.graphHeight).toBe(mobileLayout.viewportHeight);
    expect(mobileLayout.closeIsInViewport).toBe(true);
    expect(mobileLayout.hasHorizontalOverflow).toBe(false);
    expect(mobileLayout.pointsOverlapLegend).toBe(false);
    expect(mobileLayout.smallestPointTarget).toBeGreaterThanOrEqual(24);
    expect(mobileLayout.pointsAreInCanvas).toBe(true);
    expect(mobileLayout.labelsAreInCanvas).toBe(true);
    const searchBounds = await page.locator("#publication-search").boundingBox();
    const clearBounds = await page.locator("#publication-search-clear").boundingBox();
    const closeBounds = await close.boundingBox();
    expect(searchBounds?.height).toBeGreaterThanOrEqual(44);
    expect(clearBounds?.height).toBeGreaterThanOrEqual(44);
    expect(closeBounds?.width).toBeGreaterThanOrEqual(44);
    expect(closeBounds?.height).toBeGreaterThanOrEqual(44);

    await expectNoAccessibilityViolations(page);

    const search = page.locator("#publication-search");
    await search.fill("Kevin Ma Daniele Grandi");
    const mobileMatch = page.locator(".publication-link:not(.search-hidden)").first();
    await expect(mobileMatch).toBeVisible();
    await mobileMatch.click();
    await expect(page).toHaveURL(/\?view=/);
    await expect(page.locator("#publication-detail")).toBeVisible();
    await expect(page.locator("#publication-detail-link")).toHaveAttribute(
      "href",
      SCHOLAR_URL_PATTERN,
    );
    const selectedLayout = await page.evaluate(() => {
      const commandBar = document
        .getElementById("graph-command-bar")
        ?.getBoundingClientRect();
      const detail = document
        .getElementById("publication-detail")
        ?.getBoundingClientRect();
      return {
        detailIsInViewport: Boolean(
          detail
          && detail.left >= 0
          && detail.top >= 0
          && detail.right <= innerWidth
          && detail.bottom <= innerHeight
        ),
        overlapsCommandBar: Boolean(
          commandBar && detail && commandBar.bottom > detail.top
        ),
      };
    });
    expect(selectedLayout.detailIsInViewport).toBe(true);
    expect(selectedLayout.overlapsCommandBar).toBe(false);
    await expectNoAccessibilityViolations(page);

    await close.click();
    await expect(graph).toHaveAttribute("aria-hidden", "true");
    await expect(explore).toBeVisible();
    await expect(explore).toBeFocused();
  });

  test("keeps search and details separate in short landscape", async ({
    page,
  }) => {
    for (const viewport of [
      { width: 568, height: 320 },
      { width: 844, height: 390 },
    ]) {
      await page.setViewportSize(viewport);
      await page.goto("/");
      await page.locator("#exit").click();
      await page.locator("#publication-view-map").click();
      await page.locator("#publication-search").fill("Kevin Ma Daniele Grandi");
      await page.locator(".publication-link:not(.search-hidden)").first().click();

      const shortLayout = await page.evaluate(() => {
        const commandBar = document
          .getElementById("graph-command-bar")
          ?.getBoundingClientRect();
        const detailElement = document.getElementById("publication-detail");
        const detail = detailElement?.getBoundingClientRect();
        const detailBody = document.getElementById("publication-detail-body");
        return {
          detailIsInViewport: Boolean(
            detail
            && detail.left >= 0
            && detail.top >= 0
            && detail.right <= innerWidth
            && detail.bottom <= innerHeight
          ),
          detailIsScrollable: Boolean(
            detailBody
            && detailBody.scrollHeight > detailBody.clientHeight
          ),
          hasHorizontalOverflow:
            document.documentElement.scrollWidth > document.documentElement.clientWidth,
          legendIsHidden:
            getComputedStyle(document.querySelector(".graph-legends") as Element)
              .display === "none",
          overlapsCommandBar: Boolean(
            commandBar && detail && commandBar.bottom > detail.top
          ),
        };
      });

      expect(shortLayout.detailIsInViewport).toBe(true);
      expect(shortLayout.detailIsScrollable).toBe(true);
      expect(shortLayout.hasHorizontalOverflow).toBe(false);
      expect(shortLayout.legendIsHidden).toBe(true);
      expect(shortLayout.overlapsCommandBar).toBe(false);
    }
  });
});

test.describe("custom 404", () => {
  test("serves nested paths with semantic, self-hosted assets", async ({
    page,
  }) => {
    const externalResourceUrls: string[] = [];
    page.on("request", (request) => {
      const url = new URL(request.url());
      if (
        request.resourceType() !== "document" &&
        url.origin !== TEST_ORIGIN
      ) {
        externalResourceUrls.push(request.url());
      }
    });

    const response = await page.goto("/missing/nested/publication");
    expect(response?.status()).toBe(404);
    await expect(
      page.getByRole("heading", { level: 1, name: "404: Page not found" }),
    ).toBeVisible();
    await expect(page.locator('meta[name="robots"]')).toHaveAttribute(
      "content",
      "noindex, nofollow",
    );
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
      "href",
      `${SITE_ORIGIN}/404.html`,
    );

    const stylesheetUrls = await page
      .locator('link[rel="stylesheet"]')
      .evaluateAll((links) => links.map((link) => (link as HTMLLinkElement).href));
    expect(stylesheetUrls).toEqual(
      expect.arrayContaining([
        `${TEST_ORIGIN}/assets/vendor/bootstrap/bootstrap.min.css`,
        `${TEST_ORIGIN}/assets/css/default_style.css`,
      ]),
    );
    expect(stylesheetUrls.every((url) => url.startsWith(`${TEST_ORIGIN}/`))).toBe(
      true,
    );
    await expect(page.locator("script")).toHaveCount(0);
    expect(externalResourceUrls).toEqual([]);

    const homeLink = page.getByRole("link", { name: /take me home/i });
    await expect(homeLink).toHaveAttribute("href", "/");
    await expectNoAccessibilityViolations(page);
  });
});

test("publishes discovery files without development artifacts", async ({
  request,
}) => {
  const robots = await request.get("/robots.txt");
  expect(robots.status()).toBe(200);
  expect(await robots.text()).toContain(
    `Sitemap: ${SITE_ORIGIN}/sitemap.xml`,
  );

  const sitemap = await request.get("/sitemap.xml");
  expect(sitemap.status()).toBe(200);
  expect(await sitemap.text()).toContain(`<loc>${SITE_ORIGIN}/</loc>`);

  for (const path of [
    "/package.json",
    "/playwright.config.ts",
    "/tests/browser/site.spec.ts",
  ]) {
    expect((await request.get(path)).status()).toBe(404);
  }
});
