import puppeteer from 'puppeteer-core';import fs from 'node:fs';
const root='/home/hj/Alican/sites/camera-correction-results';const browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage']});
try{const p=await browser.newPage();await p.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);await p.setViewport({width:1440,height:1000});await p.goto('http://127.0.0.1:8882',{waitUntil:'networkidle0'});
 await p.$eval('#workspace',x=>x.scrollIntoView({behavior:'instant'}));await p.screenshot({path:root+'/explorer-desktop.png'});
 await p.click('[data-preset="lingbot"]');await p.waitForFunction(()=>current.model==='lingbot'&&current.seed===107&&current.scene==='02267acf');
 await p.click('[data-preset="matrixgame"]');await p.waitForFunction(()=>current.model==='matrixgame'&&current.scene==='0238de14');
 await p.evaluate(()=>{location.hash='units'});await p.waitForFunction(()=>document.getElementById('units').closest('details').open);
 await p.$eval('#studies',x=>x.scrollIntoView({behavior:'instant'}));await p.screenshot({path:root+'/studies-desktop.png'});
 await p.$eval('#diagnostics',x=>x.closest('details').open=true);const study=await p.$('.study summary');await study.click();const opened=await p.$eval('.study',x=>x.open);if(!opened)throw Error('Study disclosure failed');
 await p.$eval('.recentComparisonList summary',x=>x.click());await p.select('#nov3Model','matrixgame');await p.select('#nov3Outcome','higher');const result=await p.$eval('#nov3Table',x=>x.textContent);if(!result.includes('28.1% higher')||!result.includes('0238de14'))throw Error('Real filter interactions failed');
 await p.$eval('#nov3',x=>x.scrollIntoView({behavior:'instant'}));await p.screenshot({path:root+'/recent-results-desktop.png'});
 await p.setViewport({width:390,height:844});await p.$eval('#workspace',x=>x.scrollIntoView({behavior:'instant'}));await p.screenshot({path:root+'/explorer-mobile.png'});
 const checks=await p.evaluate(()=>({navFits:document.querySelector('.researchNav').scrollWidth<=document.querySelector('.researchNav').clientWidth,mobileFlowVisible:getComputedStyle(document.querySelector('.methodMobile')).display!=='none',noPageOverflow:document.documentElement.scrollWidth<=innerWidth,primaryPairs:EVAL_DATA.pairs.filter(x=>x.cohort==='memory_repeated').length}));if(!checks.navFits||!checks.mobileFlowVisible||!checks.noPageOverflow)throw Error(JSON.stringify(checks));
 fs.writeFileSync(root+'/qa-clean-layout.json',JSON.stringify({passed:true,...checks,studyDisclosure:true,deepLinkOpensParent:true,presetSelection:true,filterRegression:true},null,2));console.log(JSON.stringify(checks));
}finally{await browser.close()}
