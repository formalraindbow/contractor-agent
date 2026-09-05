// Run with Playwright installed: NODE_PATH=... node scripts/record_review_demo.cjs
// This makes real requests to the local agent. Each pass uses a fresh browser profile.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const base=process.env.DEMO_URL || 'http://127.0.0.1:8084';
const output=path.resolve('deliverables/interim/develop-review');
(async()=>{
 fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const first=Number(process.env.DEMO_FIRST || 1);
 const runs=first>1 ? JSON.parse(fs.readFileSync(path.join(output,"demo-runs.json"))) : [];
 for(let pass=first;pass<=3;pass++) {
  const context=await browser.newContext({viewport:{width:1440,height:1000},...(pass===1 ? {recordVideo:{dir:output,size:{width:1440,height:1000}}} : {})});
  const page=await context.newPage();const record={pass,started:new Date().toISOString(),surface:'Local Chrome, fresh profile',turns:[],errors:[]};page.on('pageerror',e=>record.errors.push(e.message));
  await page.goto(base);await page.getByRole('button',{name:'МАКСМАРКЕТ',exact:true}).click();await page.locator('#company').waitFor({state:'visible'});
  record.card=await page.locator('#card').innerText();
  assert.match(record.card,/В отчёте есть факты, требующие особого внимания/);
  await page.getByRole('button',{name:'Плачу по счёту',exact:true}).click();
  if(pass===1) await page.screenshot({path:path.join(output,'card.png')});
  for(const question of ['А сколько у них сейчас висит долгов у приставов?','С кем они судились в 2025 году и за что?']) {
   const start=Date.now();
   await page.locator('#ask').fill(question);await page.locator('#ask').press('Enter');
   await page.waitForFunction(()=>document.querySelector('#askbtn').disabled);
   await page.waitForFunction(()=>!document.querySelector('#askbtn').disabled,{},{timeout:150000});
   assert.equal(await page.locator('#thread .error-message').count(),0);
   const answer=await page.evaluate(()=>JSON.parse(localStorage.getItem('kontragent.chats.v1'))[0].turns.filter(t=>t.who==='agent').at(-1).answer);
   record.turns.push({question,seconds:(Date.now()-start)/1000,answer});
   await page.waitForFunction(()=>!document.querySelector('#askbtn').disabled);
   assert.equal(await page.locator('#card .verdict').count(),1);
   if(/приставов/.test(question)) {assert.match(answer.text_md,/54/);assert.match(answer.text_md,/453/);assert.match(answer.text_md,/11/);}
   else {assert.match(answer.text_md,/2025/);assert.match(answer.text_md,/ответчик/i);assert.match(answer.text_md,/предмет|за что|детал/);}
   if(pass===1) {await page.locator('#thread .turn').last().scrollIntoViewIfNeeded();await page.screenshot({path:path.join(output,/приставов/.test(question)?'enforcement.png':'court-2025.png')});await page.waitForTimeout(1500);}
  }
  if(pass===1) {
   await page.locator('#thread .answer').last().getByRole('button',{name:'Открыть отчёт',exact:true}).click();
   await page.getByRole('heading',{name:'Сведения о компании',exact:true}).waitFor();
   await page.screenshot({path:path.join(output,'evidence.png')});await page.waitForTimeout(1500);await page.keyboard.press('Escape');
  }
  record.status='passed';runs.push(record);const video=page.video();await context.close();
  if(video) {await video.saveAs(path.join(output,'demo.webm'));const raw=await video.path();if(raw!==path.join(output,'demo.webm')) fs.unlinkSync(raw);}
  fs.writeFileSync(path.join(output,'demo-runs.json'),JSON.stringify(runs,null,2));console.log(`Pass ${pass}: ${record.turns.map(t=>t.seconds.toFixed(1)+'s').join(', ')}; JS errors ${record.errors.length}`);
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
