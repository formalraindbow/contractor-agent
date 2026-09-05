const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const settle = page => page.evaluate(async () => {
  await Promise.allSettled(document.getAnimations()
    .filter(a => a.effect.getTiming().iterations !== Infinity).map(a => a.finished));
  let previous = -1, still = 0;
  for (let frame = 0; frame < 120 && still < 6; frame++) {
    await new Promise(requestAnimationFrame);
    const root = innerWidth <= 720 ? document.scrollingElement : document.querySelector('#workspace');
    still = Math.abs(root.scrollTop - previous) < 1 ? still + 1 : 0;
    previous = root.scrollTop;
  }
});

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:8084');
    await page.getByRole('button', { name: 'МАКСМАРКЕТ', exact: true }).click();
    await page.locator('#company').waitFor({ state: 'visible' });
    await settle(page);

    const details = page.locator('.more-facts');
    await details.locator('summary').focus();
    await page.keyboard.press('Enter');
    await settle(page);
    assert.ok(await details.evaluate(e => e.open));
    await page.keyboard.press('Space');
    await settle(page);
    assert.ok(!(await details.evaluate(e => e.open)));
    await details.locator('summary').evaluate(e => { e.click(); e.click(); e.click(); });
    await settle(page);
    assert.ok(await details.evaluate(e => e.open));
    assert.equal(await details.getAttribute('class'), 'more-facts');
    await page.getByRole('button', { name: 'Даю отсрочку', exact: true }).evaluate(e => e.click());
    await page.getByRole('button', { name: 'Плачу по счёту', exact: true }).evaluate(e => e.click());
    await settle(page);
    assert.equal(await page.locator('.motion-sizing').count(), 0);

    await page.locator('#openReport').click();
    await page.getByRole('heading', { name: 'Сведения о компании', exact: true }).waitFor();
    await page.keyboard.press('Escape');
    await page.locator('#openReport').evaluate(e => e.click());
    await page.getByRole('heading', { name: 'Сведения о компании', exact: true }).waitFor();
    await settle(page);
    assert.equal(await page.locator('#overlay').getAttribute('class'), 'overlay is-open');
    await page.keyboard.press('Escape');
    await page.locator('#overlay').waitFor({ state: 'hidden' });
    assert.equal(await page.locator('.shell').evaluate(e => e.inert), false);

    let release;
    const responseReady = new Promise(resolve => { release = resolve; });
    const answer = { kind: 'answer', text_md: '### Что означает недостоверный адрес\n\nВ отчёте есть отметка ФНС о недостоверности адресных сведений.\n\nПо указанному адресу компании может не быть.\n\nЭта отметка сама по себе не означает прекращение деятельности.\n\nОтчёт от 31.07.2026.', report_dates: { '5032257375': '2026-07-31' }, citations: [] };
    await page.route('**/v1/runs/stream', async route => {
      await responseReady;
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: [
        { type: 'token', data: { text: 'UNVALIDATED_DRAFT' } },
        { type: 'end', data: { output: answer } },
      ].map(frame => 'data: ' + JSON.stringify(frame) + '\n\n').join('') });
    });
    await page.locator('#ask').fill('Что означает недостоверный адрес?');
    await page.locator('#askbtn').click();
    await page.locator('#thread .status').waitFor({ state: 'visible' });
    await page.evaluate(() => {
      window.motionSamples = [];
      const sample = () => {
        window.motionSamples.push(document.querySelector('#thread .answer').getBoundingClientRect().height);
        if (window.motionSamples.length < 70) requestAnimationFrame(sample);
      };
      requestAnimationFrame(sample);
    });
    release();
    await page.waitForFunction(() => !document.querySelector('#askbtn').disabled);
    await settle(page);
    await page.waitForFunction(() => window.motionSamples.length === 70);
    const heights = await page.evaluate(() => window.motionSamples);
    assert.ok(heights.some(h => h > heights[0] + 3 && h < heights.at(-1) - 3), 'Reply height should grow through intermediate sizes');
    assert.doesNotMatch(await page.locator('#thread').innerText(), /UNVALIDATED_DRAFT/);
    assert.equal(await page.locator('.motion-sizing').count(), 0);
    await page.reload();
    await page.locator('#thread .answer .body').waitFor({ state: 'visible' });
    assert.match(await page.locator('#thread .answer .body').innerText(), /недостоверности адресных сведений/);

    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#thread').scrollIntoViewIfNeeded();
    await settle(page);
    await page.screenshot({ path: '/tmp/contractor-motion-mobile.png' });
    await page.locator('#navToggle').click();
    await settle(page);
    assert.equal(await page.locator('#sidebarContent').evaluate(e => e.inert), false);
    await page.keyboard.press('Escape');
    await settle(page);
    assert.equal(await page.locator('#sidebarContent').evaluate(e => e.inert), true);
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.waitForFunction(() => !document.querySelector('#sidebarContent').inert);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.getByRole('button', { name: 'Сравнить компании', exact: true }).click();
    assert.equal(await page.evaluate(() => document.getAnimations().length), 0);
    await page.getByRole('button', { name: 'Проверка компании', exact: true }).click();
    await page.getByRole('button', { name: '＋ Новая проверка', exact: true }).click();
    await page.getByRole('button', { name: 'МАКСМАРКЕТ', exact: true }).click();
    await page.locator('#company').waitFor({ state: 'visible' });
    await page.locator('.more-facts summary').click();
    assert.equal(await page.locator('.motion-sizing').count(), 0);
    assert.equal(await page.evaluate(() => document.getAnimations().length), 0);
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log('PASS: keyboard disclosures, interrupted animations, drawer reopen/focus, smooth reply growth, validated text only, history, responsive menu, reduced motion.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
