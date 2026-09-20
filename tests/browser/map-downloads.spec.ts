import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

for (const width of [375, 1280]) {
  test(`map downloads are accessible and fit a ${width}px viewport`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/assets/maps/");
    await expect(page.getByRole("heading", { name: "Publication maps", exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { level: 2 })).toHaveCount(6);
    await expect(page.locator(".variant img")).toHaveCount(12);
    await expect(page.locator(".links a[download]")).toHaveCount(36);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const violations = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
    expect(violations.violations).toEqual([]);
    const downloadEvent = page.waitForEvent("download");
    await page.getByRole("link", { name: "Download all formats · ZIP" }).click();
    const download = await downloadEvent;
    expect(download.suggestedFilename()).toBe("publication-maps.zip");
    expect(await download.failure()).toBeNull();
  });
}
