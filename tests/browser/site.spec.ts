import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const SITE_ORIGIN = "https://cmccomb.com";
const TEST_ORIGIN = "http://127.0.0.1:4173";
const SCHOLAR_URL_PATTERN = /^https:\/\/scholar\.google\.com\/citations\?/;
const HEADSHOT_URL = `${SITE_ORIGIN}/assets/images/headshot_optimized_square.jpg`;
const CMU_PROFILE_URL =
  "https://meche.engineering.cmu.edu/directory/bios/mccomb-christopher.html";

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

  test("activates the graph, supports native keyboard links, and restores focus", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    const graph = page.locator("#graph-container");
    const explore = page.locator("#exit");
    const close = page.locator("#graph-close");

    await expect(explore).toHaveAttribute("aria-expanded", "false");
    await expect(page.locator("a.publication-link")).not.toHaveCount(0);

    await explore.click();
    await expect(graph).toHaveAttribute("aria-hidden", "false");
    await expect(graph).not.toHaveAttribute("inert", "");
    await expect(explore).toHaveAttribute("aria-expanded", "true");
    await expect(close).toBeVisible();
    await expect(close).toBeFocused();
    await expectNoAccessibilityViolations(page);

    const firstPublication = page.locator("a.publication-link").first();
    const secondPublication = page.locator("a.publication-link").nth(1);
    await expect(firstPublication).toHaveAttribute("tabindex", "0");
    await expect(secondPublication).toHaveAttribute("tabindex", "-1");
    await firstPublication.focus();
    await expect(firstPublication).toBeFocused();

    await firstPublication.press("ArrowRight");
    await expect(secondPublication).toBeFocused();
    await expect(secondPublication).toHaveAttribute(
      "href",
      SCHOLAR_URL_PATTERN,
    );

    await page.context().route("https://scholar.google.com/**", async (route) => {
      await route.fulfill({
        contentType: "text/html",
        body: "<title>Google Scholar</title>",
      });
    });
    const popupPromise = page.waitForEvent("popup");
    await secondPublication.press("Enter");
    const popup = await popupPromise;
    await expect(popup).toHaveURL(SCHOLAR_URL_PATTERN);
    await popup.close();

    await close.click();
    await expect(graph).toHaveAttribute("aria-hidden", "true");
    await expect(graph).toHaveAttribute("inert", "");
    await expect(explore).toHaveAttribute("aria-expanded", "false");
    await expect(explore).toBeFocused();

    await explore.click();
    await expect(close).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(graph).toHaveAttribute("aria-hidden", "true");
    await expect(explore).toBeFocused();
  });

  test("uses the explicit Google Scholar fallback on mobile", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    const explore = page.locator("#exit");
    await expect(explore).toHaveAttribute("href", SCHOLAR_URL_PATTERN);
    await expect(explore.locator(".explore-label--scholar")).toBeVisible();
    await expect(explore.locator(".explore-label--graph")).toBeHidden();

    await page.route("https://scholar.google.com/**", async (route) => {
      await route.fulfill({
        contentType: "text/html",
        body: "<title>Google Scholar</title>",
      });
    });
    await explore.click();
    await expect(page).toHaveURL(SCHOLAR_URL_PATTERN);
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
