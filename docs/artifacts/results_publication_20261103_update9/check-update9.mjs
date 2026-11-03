import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const root='/home/hj/Alican/sites/camera-correction-results';
const browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage']});
try{
 const page=await browser.newPage(),errors=[],bad=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)bad.push([r.status(),r.url()])});
 await page.setViewport({width:1440,height:1000});await page.goto('http://127.0.0.1:8882',{waitUntil:'networkidle0'});
 const initial=await page.evaluate(()=>({model:document.querySelector('#model').value,cohort:document.querySelector('#cohort').value,catalog:document.querySelectorAll('.study').length,mode:document.querySelector('#modeTitle').textContent}));
 console.log(JSON.stringify({initial,errors}));
 for(const mode of ['L','A','C','M','B']){
  await page.click(`[data-mode="${mode}"]`);const active=await page.$eval(`[data-mode="${mode}"]`,e=>e.getAttribute('aria-pressed'));if(active!=='true')throw Error('Method control failed '+mode);
 }
 await page.click('[data-mode="L"]');
 await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));await page.screenshot({path:root+'/overview-desktop.png',fullPage:false});
 const result=await page.evaluate(()=>{
  const assert=(x,msg)=>{if(!x)throw Error(msg)};const el=id=>document.getElementById(id);
  let checked=0,invalid=0,confounded=0;
  for(const pair of EVAL_DATA.pairs){
   el('model').value=pair.model;modelChanged();el('cohort').value=pair.cohort;cohortChanged();el('seed').value=String(pair.seed);seedChanged();el('scene').value=pair.id;selectPair();seek(pair.horizon);
   assert(current.id===pair.id,'Selection '+pair.id);
   assert(el('baselineLabel').textContent===(pair.left_label||'Baseline'),'Left label '+pair.id);
   assert(!el('results').hidden,'Missing results '+pair.id);
   if(pair.position_valid===false){invalid++;assert(el('poseMode').value==='orientation','Invalid position plotted');assert(el('errorMetric').value==='rotation_errors','Invalid position error plotted')}
   if(pair.comparison_valid===false||pair.comparison_valid===null){confounded++;assert(!el('comparisonWarning').hidden,'Confound warning missing');assert(el('takeaway').textContent.includes('unavailable'),'Confound aggregate populated')}
   if(pair.experiment==='memory')assert(el('metricWindow').textContent.includes('Post-intervention'),'Memory window missing');
   checked++;
  }
  assert(EVAL_DATA.pairs.length===203,'Comparison count');const future=EVAL_DATA.pairs.filter(p=>p.experiment==='dreamx_future');assert(future.length===4,'DreamX cases');assert(document.querySelectorAll('#dreamxFutureRows tr').length===4,'DreamX table');assert(document.querySelectorAll('#dreamxFutureGallery img').length===4,'DreamX sheets');assert(future.some(p=>p.camera_change_pct<0),'DreamX negative result missing');assert(el('dreamxFutureLimits').textContent.includes('color-corrected'),'RGB confound missing');for(const p of future){inspectResearchPair(p.id);assert(el('metricWindow').textContent.includes('33–68'),'DreamX scoring window');assert(p.arms.correction.feedback.changed_native_chunks===3,'DreamX actuation');}assert(document.querySelectorAll('#nov3Table tr').length===24,'New result rows');assert(document.querySelectorAll('#nov3Gallery img').length===24,'Reference sheets');const ling=EVAL_DATA.pairs.find(p=>p.model==='lingbot'&&p.cohort==='memory_repeated');inspectResearchPair(ling.id);assert(!el('comparisonWarning').hidden&&el('comparisonWarning').textContent.includes('collapse'),'Visual failure warning');assert(eligible(ling),'Visual failure silently dropped');const failures=EVAL_DATA.pairs.filter(p=>p.cohort==='memory_repeated'&&/failure|collapse/i.test(p.visual_classification||''));assert(failures.length===3,'Fidelity failures missing');for(const p of failures){inspectResearchPair(p.id);assert(!el('comparisonWarning').hidden,'Fidelity warning missing');assert(eligible(p),'Fidelity failure excluded');}assert(el('liveStatus').textContent.includes('24/40'),'Coverage summary stale');assert(Object.keys(EVAL_DATA.research.nov3.queues).length===15,'Queues missing');const missing=EVAL_DATA.research.memory.rows.filter(x=>x.state!=='completed');assert(missing.length===2,'Missing failures');
  const abstained=EVAL_DATA.pairs.filter(p=>p.cohort==='memory_repeated'&&p.accepted_memory_writes===0);assert(abstained.length===6,'Abstention count');for(const p of abstained){assert(p.full_rgb_exact===true,'Missing full RGB equality');assert(p.position_valid===false,'Abstention position silently valid');assert(p.additional_visual_evidence.some(x=>x.label.includes('RGB equality')),'Missing RGB receipt link');}const allframes=EVAL_DATA.pairs.find(p=>p.cohort==='memory_repeated'&&p.model==='lingbot'&&p.scene==='0238de14'&&p.seed===107);assert(allframes.additional_visual_evidence.length===2,'Missing all-frame review links');assert(el('nov3Queues').textContent.includes('LingBot normalization'),'Normalization queue missing');
  assert(el('gainStats').textContent.includes('1/8'),'Activation count');
  assert(el('responseReplaySummary').textContent.includes('2/8'),'Separate RE10K activation audit missing');
  assert(document.querySelectorAll('#responseReplayRows tr').length===8,'RE10K audit rows missing');
  const canonical=EVAL_DATA.pairs.filter(p=>p.cohort==='shared_bc'&&p.identical_observer_inputs);
  assert(canonical.length===6,'Six exact-output comparisons missing');
  for(const p of canonical){assert(JSON.stringify(p.arms.baseline.metrics)===JSON.stringify(p.arms.correction.metrics),'Canonical tie missing');assert(p.original_observer_evaluation,'Raw evaluation provenance missing');inspectResearchPair(p.id);assert(el('comparisonWarning').textContent.includes('no control effect'),'Exact-output interpretation missing');}
  assert(el('zigzagFinding').textContent.includes('25 accepted'),'Post-collapse feedback missing');

  assert(document.querySelectorAll('.study').length===25,'Catalog count');
  assert(document.querySelectorAll('details.study:not([open])').length===25,'Study details start closed');
  assert(document.querySelectorAll('#cohortOverview tr').length===5,'Overview models');
  assert(el('decoderFinding').textContent.includes('46.6–91.5%'),'Decoder result missing');
  assert(el('nov3Queues').textContent.includes('RGB equality failed'),'WorldPlay failure missing');
  el('nov3Scope').value='all';el('nov3Scope').dispatchEvent(new Event('change'));assert(document.querySelectorAll('#nov3Table tr').length===33,'All recent comparisons');
  el('nov3Scope').value='primary';el('nov3Scope').dispatchEvent(new Event('change'));
  el('nov3Model').value='matrixgame';el('nov3Model').dispatchEvent(new Event('change'));assert(document.querySelectorAll('#nov3Table tr').length===8,'Matrix recent filter');
  el('nov3Outcome').value='higher';el('nov3Outcome').dispatchEvent(new Event('change'));assert(document.querySelectorAll('#nov3Table tr').length===1&&el('nov3Table').textContent.includes('0238de14'),'Regression filter');
  el('nov3Model').value='all';el('nov3Outcome').value='all';el('nov3Model').dispatchEvent(new Event('change'));
  for(const id of ['ablations','diagnostics','units'])assert(el(id).closest('details')&&!el(id).closest('details').open,'Detail hierarchy '+id);

  assert(el('legacyFailureCount').textContent.includes('21/79'),'Failure denominator missing');
  assert(document.querySelectorAll('[data-failure-pair]').length===8,'Eight model findings missing');
  assert(document.querySelectorAll('#vmemArmRows tr').length===4,'VMem factorial arms missing');
  assert(el('vmemArmRows').textContent.includes('+93.7%')&&el('vmemArmRows').textContent.includes('-25.1%'),'VMem command/memory attribution missing');
  assert(el('legacyFailureCards').textContent.includes('zero adjacent intended translation reversals'),'DreamX correction missing');
  assert(el('vmemReplaySummary').textContent.includes('5/10')&&el('vmemReplaySummary').textContent.includes('0.01632'),'VMem incremental effect missing');
  assert(document.querySelectorAll('#vmemReplayNotes li').length===3,'Replay qualifications missing');
  assert(el('vmemReplayNotes').textContent.includes('per-window camera normalization')&&el('vmemReplayNotes').textContent.includes('not native amplitudes'),'Scale qualification missing');
  assert(document.querySelectorAll('#vmemReplaySources a').length===5,'Replay evidence links missing');
  const resumes=Object.values(EVAL_DATA.research.nov3.queues).filter(q=>q.previous_attempts.length);assert(resumes.length===10&&resumes.every(q=>q.previous_attempts.length===2),'Both earlier holds must remain visible');
  assert(el('nov3Queues').textContent.includes('earlier hold 2'),'Second hold link missing');
  assert(Object.values(EVAL_DATA.research.nov3.queues).filter(q=>q.previous_attempt&&q.process_live_at_export).length===8,'Resume provenance missing');
  for(const b of document.querySelectorAll('[data-failure-pair]')){b.click();assert(current.id===b.dataset.failurePair,'Failure case navigation');assert(!el('comparisonWarning').hidden,'Failure interpretation missing in explorer');}
  const latest=EVAL_DATA.pairs.find(p=>p.cohort==='memory_repeated'&&p.model==='matrixgame'&&p.scene==='0238de14'&&p.seed===107);inspectResearchPair(latest.id);assert(el('comparisonWarning').textContent.includes('worse endpoint position and rotation'),'New tradeoff warning missing');
  inspectResearchPair('vmem_re10k40_d3f042a2_s7');assert(el('comparisonWarning').textContent.includes('not a command-only'),'VMem policy interpretation missing');
  assert(document.querySelectorAll('#legacyRejoinRows tr').length===10,'Legacy cohorts missing');assert(document.getElementById('legacySource').textContent.includes('80 source records'),'Legacy audit missing');for(const b of document.querySelectorAll('[data-legacy-pair]')){b.click();assert(current.id===b.dataset.legacyPair,'Legacy evidence link missing');}assert(!document.body.textContent.includes('[object Object]'),'Object rendered');
  el('probeObserver').value='vipe';el('probeObserver').dispatchEvent(new Event('change'));assert(el('probeChart').querySelectorAll('circle').length===8,'Probe observer switch');
  el('probeObserver').value='both';el('probeObserver').dispatchEvent(new Event('change'));
  inspectResearchPair(EVAL_DATA.pairs.find(p=>p.cohort==='abc_AB'&&p.model==='lingbot'&&p.seed===7&&p.scene==='0238de14').id);
  return {comparisons:checked,invalidPosition:invalid,unqualifiedComparisons:confounded,studies:25};
 });
 await page.waitForFunction(()=>document.getElementById('baselineVideo').readyState>=1&&document.getElementById('correctionVideo').readyState>=1);
 await page.evaluate(()=>{seek(0)});await page.click('#play');await new Promise(r=>setTimeout(r,1500));await page.click('#play');
 const playback=await page.evaluate(()=>({left:document.getElementById('baselineVideo').currentTime,right:document.getElementById('correctionVideo').currentTime}));if(Math.abs(playback.left-playback.right)>.3||playback.left<.1)throw Error('Video synchronization failed');
 await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));await page.screenshot({path:root+'/desktop.png'});
 await page.$eval('#nov3',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/nov3-desktop.png'});await page.$eval('#method',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/method-desktop.png'});
 await page.$eval('#legacyFailures',e=>{e.open=true;e.scrollIntoView({behavior:'instant'})});await page.screenshot({path:root+'/failure-desktop.png'});await page.$eval('#vmemReplay',e=>{e.open=true});await page.$eval('.vmemArms',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/vmem-desktop.png'});
 await page.$eval('#ablations',e=>{e.closest('details').open=true});await page.$eval('#responseReplay',e=>{e.open=true;e.scrollIntoView({behavior:'instant'})});await page.screenshot({path:root+'/response-audit-desktop.png'});
 await page.$eval('#dreamxFutureResults',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/dreamx-future-desktop.png'});await page.setViewport({width:390,height:844});await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));await page.screenshot({path:root+'/mobile.png'});
 const overflow=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,width:innerWidth}));if(overflow.scroll>overflow.width+1)throw Error('Mobile overflow '+JSON.stringify(overflow));
 await page.$eval('#responseReplay',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/response-audit-mobile.png'});
 await page.$eval('#vmemReplay',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/vmem-mobile.png'});await page.$eval('#method',e=>e.scrollIntoView({behavior:'instant'}));await page.click('[data-mode="M"]');await page.screenshot({path:root+'/method-mobile.png'});
 await page.$eval('#dreamxFutureResults',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/dreamx-future-mobile.png'});if(errors.length)throw Error(errors.join('\n'));if(bad.length)throw Error(JSON.stringify(bad.slice(0,10)));
 await page.$eval('#nov3',e=>e.scrollIntoView({behavior:'instant'}));await page.screenshot({path:root+'/nov3-mobile.png'});const out={passed:true,...result,initial,playback,overflow,errors,httpErrors:bad};fs.writeFileSync(root+'/qa-update9-browser.json',JSON.stringify(out,null,2));console.log(JSON.stringify(out));
}finally{await browser.close()}
