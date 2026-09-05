const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[]; page.on('pageerror',e=>errors.push(e.message));
    await page.goto('http://127.0.0.1:8084');
    await page.locator('#q').fill('7712345678');
    await page.locator('#q').press('Enter');
    await page.getByText('Компании с ИНН 7712345678 нет в демобазе. В ней 200 компаний.',{exact:true}).waitFor();
    await page.route('**/report/7712345678/card',route=>route.fulfill({status:503,body:'Unavailable'}));
    await page.locator('#q').fill('7712345678'); await page.locator('#q').press('Enter');
    await page.getByText('Связь со стендом прервалась. Обновите страницу и попробуйте ещё раз.',{exact:true}).waitFor();
    await page.unroute('**/report/7712345678/card');
    await page.getByRole('button',{name:'ГДК',exact:true}).click();
    await page.locator('#company').waitFor({state:'visible'});
    assert.equal(await page.locator('#card .assessment-label').textContent(),'Что важно в отчёте');
    await page.locator('#openReport').click();
    await page.getByRole('heading',{name:'Сведения о компании',exact:true}).waitFor();
    assert.match(await page.locator('#drawerContent tr').filter({hasText:'Статус'}).textContent(),/Действующая/);
    assert.doesNotMatch(await page.locator('#drawerContent').innerText(),/Основание статуса не указано/);
    await page.keyboard.press('Escape');

    let finish,request;
    const gate = new Promise(resolve=>finish=resolve);
    const raw='Рекомендация помощника: Есть существенные риски.\n\nСветофор — красный (HIGH). Блокировка счетов (moderate).\n\nВ отчёте нет финансовой отчётности (раздел **finReports** не представлен).\n\nЛиквидность: 0.49 [, ].\n\nУбыток за 2024 год ‑ 26,2 млн ₽. Нет данных о оборотных активах.';
    await page.route('**/v1/runs/stream',async route=>{
      request=route.request().postDataJSON(); await gate;
      await route.fulfill({contentType:'text/event-stream',body:'data: '+JSON.stringify({type:'end',data:{output:{kind:'answer',text_md:raw,report_dates:{'6165169320':'2026-08-25'},citations:[]}}})+'\n\n'});
    });
    await page.locator('#ask').fill('Покажи финансы по годам'); await page.locator('#ask').press('Enter');
    await page.waitForFunction(()=>document.querySelector('#askbtn').disabled);
    assert.equal(await page.locator('#ask').inputValue(),'');
    const draft='Следующий вопрос о компании';
    await page.locator('#ask').pressSequentially(draft,{delay:10});
    await page.locator('#ask').press('Enter');
    assert.equal(await page.locator('#ask').inputValue(),draft);
    finish(); await page.waitForFunction(()=>!document.querySelector('#askbtn').disabled);
    assert.equal(await page.locator('#ask').inputValue(),draft);
    assert.equal(await page.locator('#thread .turn.user').count(),1);
    assert.match(request.input.messages[0].content,/Покажи финансы по годам/);
    const visible=await page.locator('#thread .answer .body').innerText();
    assert.match(visible,/Что важно в отчёте: В отчёте есть факты, требующие особого внимания/);
    assert.equal(await page.evaluate(()=>cleanText('С компанией можно работать')),'Можно работать');
    assert.equal(await page.evaluate(()=>cleanText('В отчёте отмечено: **«в отчёте есть факты, требующие особого внимания»**')),'В отчёте есть факты, требующие особого внимания.');
    assert.doesNotMatch(visible,/Есть существенные риски|Рекомендация помощника|HIGH|moderate|finReports|\[,|о оборотных/);
    assert.match(visible,/год — 26,2/);
    await page.context().grantPermissions(['clipboard-read','clipboard-write']);
    await page.locator('#thread .copy-answer').click();
    assert.doesNotMatch(await page.evaluate(()=>navigator.clipboard.readText()),/Есть существенные риски|Рекомендация помощника|HIGH|moderate|finReports|\[,/);
    await page.reload(); await page.locator('#thread .answer .body').waitFor();
    assert.doesNotMatch(await page.locator('#thread .answer .body').innerText(),/Есть существенные риски|Рекомендация помощника|HIGH|moderate|finReports|\[,/);
    assert.equal(await page.locator('#ask').inputValue(),'');
    await page.locator('#ask').fill('Черновик ГДК');
    await page.locator('#reset').click();
    await page.getByRole('button',{name:'ЯНПОЛОВ',exact:true}).click();
    await page.locator('.bank-note').waitFor();
    assert.match(await page.locator('.bank-note').innerText(),/Светофор банка — красный/);
    assert.match(await page.locator('.bank-note').innerText(),/Причина оценки в отчёте не раскрыта/);
    assert.equal(await page.locator('#ask').inputValue(),'');
    for (const key of ['look','pay','deal','supply','legal']) {
      await page.locator('#goals [data-goal="'+key+'"]').click();
      assert.equal(await page.locator('#card .v').textContent(),'Нужна дополнительная проверка');
      assert.match(await page.locator('.bank-note').innerText(),/Светофор банка — красный/);
    }
    assert.match(await page.locator('#cardDetails').innerText(),/Что можно запросить/);
    assert.match(await page.locator('#cardDetails').innerText(),/Документы, подтверждающие требование/);
    await page.locator('#ask').fill('Черновик ИП');
    await page.locator('#chats .chat-open').filter({hasText:'ООО «ГДК»'}).click();
    await page.locator('#cname').filter({hasText:'ООО «ГДК»'}).waitFor();
    assert.equal(await page.locator('#ask').inputValue(),'');

    await page.evaluate(()=>{
      const chats=loadChats(), sample=chats[0];
      saveChats([...chats,...Array.from({length:10},(_,i)=>({...sample,id:'history-'+i,name:'Проверка '+i}))]);
      renderChats();
    });
    await page.setViewportSize({width:375,height:812});
    await page.reload(); await page.locator('#company').waitFor({state:'visible'});
    await page.evaluate(()=>{followResponse=false;window.scrollTo({top:0,behavior:'instant'});});
    assert.equal(await page.locator('#navToggle').getAttribute('aria-expanded'),'false');
    assert.ok((await page.locator('.sidebar').boundingBox()).height<100);
    assert.ok((await page.locator('#company').boundingBox()).y<180);
    assert.equal(await page.locator('#sidebarContent').isVisible(),false);
    await page.locator('#navToggle').click();
    assert.equal(await page.locator('#sidebarContent').isVisible(),true);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#sidebarContent').isVisible(),false);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    fs.mkdirSync('deliverables/interim/stand-issues',{recursive:true});
    await page.screenshot({path:'deliverables/interim/stand-issues/mobile.png'});

    await page.setViewportSize({width:1440,height:1000});
    await page.locator('#tabMany').click(); await page.locator('#cmpExample').click();
    await page.waitForFunction(()=>document.querySelectorAll('#selected .chip').length===3 && !document.querySelector('#cmpgo').disabled);
    await page.locator('#cmpgo').click();
    for(const text of await page.locator('.comparison-verdict.not_recommended').allTextContents()) assert.equal(text,'В отчёте есть факты, требующие особого внимания');
    const originalGaps=await page.evaluate(()=>chosen[1].card.gaps);
    await page.evaluate(()=>{chosen[1].card.gaps=['Раздел «Финансовая отчётность» в отчёте есть, но данных в нём нет — оценить финансы по отчёту нельзя.'];renderComparison(chosen.map(c=>c.card));});
    assert.doesNotMatch(await page.locator('#cmpBody').innerText(),/Раздел «Финансовая отчётность»/);
    await page.evaluate(gaps=>{chosen[1].card.gaps=gaps;renderComparison(chosen.map(c=>c.card));},originalGaps);
    assert.equal(errors.length,0,errors.join('\n'));
    await page.screenshot({path:'deliverables/interim/stand-issues/desktop.png'});
    console.log('PASS: missing company versus network error, actual status, typing during generation, Enter, isolated composers, historical text and clipboard cleanup, red bank explanation, invariant verdict across all five goals, relevant claim documents, collapsed mobile history, consistent comparison wording.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exit(1);});
