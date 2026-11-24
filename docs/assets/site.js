const SITE = {
  repositoryUrl: "https://github.com/computationalpathologygroup/segmenteer",
  version: "0.1.0",
};


function initRepositoryLinks() {
  document.querySelectorAll("[data-repo-link]").forEach((link) => {
    link.href = SITE.repositoryUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
  document.querySelectorAll("[data-version]").forEach((node) => {
    node.textContent = SITE.version;
  });
}

function initNavigation() {
  const header = document.querySelector("[data-header]");
  const button = document.querySelector("[data-menu-button]");
  if (!header || !button) return;

  button.addEventListener("click", () => {
    const isOpen = header.dataset.open === "true";
    header.dataset.open = String(!isOpen);
    button.setAttribute("aria-expanded", String(!isOpen));
  });

  header.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      header.dataset.open = "false";
      button.setAttribute("aria-expanded", "false");
    });
  });
}

function initBoundaryStudy() {
  const figure = document.querySelector("[data-boundary-study]");
  const svg = figure?.querySelector(".study-svg");
  if (!figure || !svg) return;

  const speckleGroup = figure.querySelector(".specimen-speckles");
  if (!speckleGroup) return;

  const fragment = document.createDocumentFragment();
  let seed = 17;
  const random = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };

  for (let i = 0; i < 90; i += 1) {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", String(115 + random() * 520));
    circle.setAttribute("cy", String(100 + random() * 400));
    circle.setAttribute("r", String(1 + random() * 4));
    fragment.appendChild(circle);
  }
  speckleGroup.appendChild(fragment);
}





initRepositoryLinks();
initNavigation();
initBoundaryStudy();



/* Display limits come from the benchmark export. */
function leaderboardDisplayLimit(benchmark) {
  const cap = benchmark?.display_top;
  return Number.isSafeInteger(cap) && cap > 0 ? cap : Infinity;
}

  // Metric arrows indicate the preferred direction.
async function initLeaderboard() {
  const table = document.querySelector('[data-leaderboard]');
  const status = document.querySelector('[data-order-status]');
  const instruction = document.querySelector('.lb-desktop-instruction');
  if (!table) return;
  const preferredDirection = { dice: 'desc', nsd: 'desc', under: 'asc', over: 'asc', speed: 'asc' };
  const metricNames = { dice: 'Dice', nsd: 'NSD', under: 'Under-segmentation', over: 'Over-segmentation', speed: 'Average time per slide' };
  const datasetNames = { development: 'Tissue-with-issues', test: 'Private hidden' };
  const body = table.tBodies[0];
  let raw;
  try {
    const response = await fetch('assets/benchmark-results.json', {cache:'no-store'});
    if (!response.ok) throw Error(`Benchmark HTTP ${response.status}`);
    raw = await response.json();
    if (!Array.isArray(raw.methods)) throw Error('Malformed benchmark data');
  } catch (error) {
    console.error('Could not load benchmark results:', error);
    if (status) status.textContent = 'Unable to load results. Serve this site over HTTP and regenerate benchmark data.';
    return;
  }
  // Use the timestamp produced by the same export that populated the rows.
  // Keep it hidden if the data does not supply a valid generation time.
  const updated = document.querySelector('[data-benchmark-updated]');
  if (updated && typeof raw.generated_at === 'string') {
    const generatedDate = new Date(raw.generated_at);
    if (Number.isFinite(generatedDate.getTime())) {
      const formatted = new Intl.DateTimeFormat('en-GB', {
        day: 'numeric', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
        timeZone: 'UTC'
      }).format(generatedDate);
      updated.dateTime = generatedDate.toISOString();
      updated.textContent = `Last updated ${formatted} UTC`;
      updated.hidden = false;
    }
  }
  const cohortKey = typeof raw.cohort_key === 'string' && raw.cohort_key.trim();
  const input = cohortKey ? raw.methods : [];
  const usedMethods = new Set();
  const validRows = input.filter(record => {
    if (!record || record.matched_slides !== raw.matched_slides || record.cohort_key !== cohortKey) return false;
    if (typeof record.method !== 'string' || !record.method.trim()) return false;
    if (!record.run_id || usedMethods.has(record.run_id)) return false;
    if (!['dice','nsd','under','over'].every(metric =>
       typeof record[metric] === 'number' && Number.isFinite(record[metric]))) return false;
    if (record.speed != null && !(typeof record.speed === 'number' && Number.isFinite(record.speed))) return false;
    usedMethods.add(record.run_id);
    return true;
  });

  const sealedCell = () => {
    const cell = document.createElement('td');
    cell.className = 'lb-sealed-cell';
    cell.title = 'Private hidden test dataset · sealed';
    cell.setAttribute('aria-label', 'Sealed');
    const dots = document.createElement('span');
    dots.className = 'lb-sealed-dots';
    dots.setAttribute('aria-hidden', 'true');
    dots.textContent = '•••';
    cell.append(dots);
    return cell;
  };
  validRows.forEach((record, index) => {
    const row = document.createElement('tr');
    const heading = document.createElement('th');
    heading.scope = 'row';
    const rank = document.createElement('span');
    rank.className = 'lb-rank';
    rank.textContent = String(index + 1).padStart(2,'0');
    const method = document.createElement('span');
    method.className = 'lb-method';
    // Support structured identities and older manifest formats.
    const name = typeof record.method_name === 'string' && record.method_name.trim()
      ? record.method_name.trim() : record.method.split('/', 1)[0].trim();
    const architectureName = typeof record.architecture === 'string' && record.architecture.trim()
      ? record.architecture.trim() : record.method.includes('/')
        ? record.method.slice(record.method.indexOf('/') + 1).trim() : '';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'lb-method-name';
    nameSpan.textContent = name;
    method.append(nameSpan);
    if (architectureName) {
      const architecture = document.createElement('span');
      architecture.className = 'lb-architecture';
      architecture.textContent = ' / ' + architectureName;
      method.append(architecture);
    }
    heading.append(rank, method);
    row.append(heading);
    ['dice','nsd','under','over','speed'].forEach(metric => {
      const cell = document.createElement('td');
      if (record[metric] == null) {
        cell.textContent = '—';
        cell.className = 'lb-unavailable';
        cell.setAttribute('aria-label', 'Not available');
      } else {
        cell.dataset.value = String(record[metric]);
        cell.textContent = metric === 'under' || metric === 'over'
          ? (record[metric] * 100).toFixed(1) + '%'
          : record[metric].toFixed(metric === 'speed' ? 2 : 3);

      }
      row.append(cell);
    });
    for (let i = 0; i < 5; i++) row.append(sealedCell());
    body.append(row);
  });

  const developmentButtons = table.querySelectorAll('button[data-sort^="development:"]');
  developmentButtons.forEach(button => {
    const metric = button.dataset.sort.split(':')[1];
    button.disabled = validRows.length === 0 || !validRows.some(record => record[metric] != null);
  });
  if (instruction) instruction.hidden = validRows.length === 0;
  if (!validRows.length) {
    if (status) status.textContent = '';
    return;
  }
  let sortState = { key: 'development:'+ (raw.ranking_metric==='nsd'?'nsd':'dice'), direction: 'desc' };
  function numericValue(row, columnIndex) {
    const value = row.cells[columnIndex].getAttribute('data-value');
    if (value === null || value.trim() === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }
  const allBenchmarkRows = [...body.rows];
  function sortTable(key, direction) {
    const header = [...table.querySelectorAll('thead th[data-sort-key]')]
      .find(th => th.dataset.sortKey === key);
    if (!header) return;
    const columnIndex = header.cellIndex + 1; // Exclude the spanning method header.
    const [, metric] = key.split(':');
    const bestDirection = preferredDirection[metric];
    const allRows = [...allBenchmarkRows];
    const compare = (a,b,order) => {
      const av = numericValue(a,columnIndex);
      const bv = numericValue(b,columnIndex);
      if (av === null) return bv === null ? 0 : 1;
      if (bv === null) return -1;
      const result = order === 'asc' ? av - bv : bv - av;
      return result || a.querySelector('.lb-method').textContent.localeCompare(b.querySelector('.lb-method').textContent);
    };
    // Apply the display limit to the highest-ranked methods.
    allRows.sort((a,b) => compare(a,b,bestDirection));
    const rankByRow = new Map(allRows.map((row,i) => [row, i+1]));
    const cap = leaderboardDisplayLimit(raw);
    const topRows = Number.isFinite(cap)
      ? allRows.filter(row => numericValue(row,columnIndex) !== null).slice(0,cap)
      : allRows;
    const shownRows = [...topRows].sort((a,b) => compare(a,b,direction));
    // Retain rows for subsequent sorting and display changes.
    body.replaceChildren(...shownRows);
    for (const row of allRows) {
      row.querySelector('.lb-rank').textContent = String(rankByRow.get(row)).padStart(2,'0');
      [...row.cells].forEach(cell => cell.classList.remove('lb-active-value'));
      row.cells[columnIndex].classList.add('lb-active-value');
    }
    // Preserve row references across DOM reordering.
    sortState = {key,direction};
    table.querySelectorAll('thead th[data-sort-key]').forEach(th => {
      th.removeAttribute('aria-sort');
      if (th.dataset.sortKey === key) th.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : 'descending');
    });
    const limitText = Number.isFinite(cap) ? `Top ${cap}` : 'All methods';
    if (status) status.textContent = `${limitText} · ${metricNames[metric]} · ${direction===bestDirection?'best first':'reverse order'}`;
  }
  table.querySelectorAll('button[data-sort]:not(:disabled)').forEach(button => {
    button.addEventListener('click', () => {
      const key = button.dataset.sort;
      const metric = key.split(':')[1];
      const direction = sortState.key === key
        ? (sortState.direction === 'asc' ? 'desc' : 'asc')
        : preferredDirection[metric];
      sortTable(key,direction);
    });
  });
  sortTable(sortState.key,sortState.direction);
}
initLeaderboard();




/* Metadata-selected case showcase */
(function initAdaptiveShowcase(){
  const root=document.querySelector('[data-gd-case]');
  if(!root)return;
  const preview=root.querySelector('[data-gd-preview]');
  const tabsRoot=root.querySelector('.gd-tabs');
  const numeric=root.querySelector('[data-gd-numeric]');
  const imagery=root.querySelector('[data-gd-imagery]');
  const original=root.querySelector('[data-gd-original]');
  const imgSlide=root.querySelector('[data-gd-slide]');
  const imgPrediction=root.querySelector('[data-gd-prediction]');
  let models=[];let current=0;let imagesReady=false;let rankingMetric='dice';let imageVersion='';
  const fmt=value=>Number.isFinite(value)?value.toFixed(3):'—';
  const percent=value=>Number.isFinite(value)?(value*100).toFixed(1)+'%':'—';
  const imagePath=name=>`assets/${encodeURIComponent(name)}${imageVersion?'?v='+imageVersion:''}`;
  function render(i){
    current=i;const model=models[i];if(!model)return;
    const m=model.metrics;
    root.querySelector('[data-gd-dice]').textContent=fmt(m[rankingMetric]);
    root.querySelector('[data-gd-nsd]').textContent=fmt(rankingMetric==='nsd'?m.dice:m.nsd);
    root.querySelector('[data-gd-under]').textContent=percent(m.under);
    root.querySelector('[data-gd-over]').textContent=percent(m.over);
    root.querySelector('[data-gd-method]').textContent=model.label;
    root.querySelector('[data-gd-bar]').style.width=(100*m[rankingMetric]).toFixed(2)+'%';
    [...tabsRoot.querySelectorAll('button')].forEach((btn,k)=>{
      btn.classList.toggle('gd-tab-active',k===i);
      btn.setAttribute('aria-pressed',String(k===i));
    });
    if(imagesReady){
      imgPrediction.src=imagePath(model.overlay);
      imgPrediction.alt=`${model.label} prediction for the selected slide`;
      root.querySelector('[data-gd-image-label]').textContent='Prediction · '+model.label;
    }
  }
  function probe(url){
    return new Promise(resolve=>{
      const im=new Image();im.onload=()=>resolve(true);im.onerror=()=>resolve(false);im.src=url;
    });
  }
  const showOriginal=active=>{
    imgPrediction.style.opacity=active?'0':'1';
    original.setAttribute('aria-pressed',String(active));
  };
  original.addEventListener('pointerdown',()=>showOriginal(true));
  ['pointerup','pointercancel','pointerleave','blur'].forEach(x=>original.addEventListener(x,()=>showOriginal(false)));
  original.addEventListener('keydown',evt=>{if(evt.key===' '||evt.key==='Enter')showOriginal(true);});
  original.addEventListener('keyup',()=>showOriginal(false));
  fetch('assets/case-showcase.json',{cache:'no-store'}).then(response=>{
    if(!response.ok)throw Error(`showcase manifest ${response.status}`);
    return response.json();
  }).then(async data=>{
    if(data.status==='unavailable' || !Array.isArray(data.models) || !data.models.length) {
      preview.textContent='Case visualization unavailable';
      root.querySelector('[data-gd-method]').textContent='No verified case available';
      return;
    }
    rankingMetric=data.ranking_metric==='nsd'?'nsd':'dice';
    // Fixed filenames are safe to cache-bust using the refreshed manifest timestamp.
    imageVersion=encodeURIComponent(data.generated_at || '');
    const sourceName=data.metadata?.source_site || data.metadata?.data_source || 'BENCHMARK';
    const datasetName=data.metadata?.data_source || 'BENCHMARK';
    const header=root.querySelector('#gd-case-title');
    if(header)header.textContent='REAL CASE / '+sourceName.toUpperCase();
    const sourceLabel=root.querySelector('.gd-topline-source');
    if(sourceLabel)sourceLabel.textContent=datasetName.toUpperCase();
    imgSlide.alt='Original whole-slide specimen · '+sourceName;
    if (typeof data.annotation_color === 'string' && /^#[0-9a-fA-F]{6}$/.test(data.annotation_color)) {
      root.style.setProperty('--gd-annotation', data.annotation_color);
    }
    models=data.models;
    const secondaryLabel=root.querySelector('[data-gd-secondary-label]');
    if(secondaryLabel)secondaryLabel.textContent=(rankingMetric==='nsd'?'DICE':'NSD')+' ↑';
    const caseMetricTitle=root.querySelector('[data-gd-score-heading]');
    if(caseMetricTitle)caseMetricTitle.textContent=rankingMetric.toUpperCase()+' SCORE · THIS SLIDE';
    const captionMetric=root.querySelector('[data-gd-caption]');
    if(captionMetric)captionMetric.textContent='Top 3 by overall '+rankingMetric.toUpperCase()+' · Scores on this slide';
    root.querySelector('.gd-specimen-name').textContent=data.metadata?.organ||'Tissue specimen';
    root.querySelector('.gd-specimen-stain').textContent=data.metadata?.staining||'';
    tabsRoot.replaceChildren();
    models.forEach((model,i)=>{
      const btn=document.createElement('button');btn.type='button';
      btn.className='gd-tab'+(i===0?' gd-tab-active':'');
      btn.setAttribute('aria-pressed',String(i===0));
      const number=document.createElement('span');number.className='gd-tab-rank';number.textContent=String(i+1).padStart(2,'0');
      const label=document.createElement('span');label.className='gd-tab-name';
      const parts=model.label.split('/');
      label.textContent=parts[0];
      if(parts.length>1){const arch=document.createElement('span');arch.className='gd-tab-arch';arch.textContent='/'+parts.slice(1).join('/');label.append(arch);}
      const score=document.createElement('b');score.textContent=fmt(model.metrics[rankingMetric]);
      btn.append(number,label,score);
      btn.addEventListener('click',()=>render(i));tabsRoot.append(btn);
    });
    render(0);
    const urls=[data.thumbnail,...models.map(m=>m.overlay)];
    if(urls.every(Boolean)&& (await Promise.all(urls.map(name=>probe(imagePath(name))))).every(Boolean)){
      imgSlide.src=imagePath(data.thumbnail);
      imagesReady=true;
      imagery.hidden=false;numeric.hidden=true;
      preview.textContent='Select a method · Hold to view original';
      render(0);
    } else {
      preview.textContent='Case scores';
    }
  }).catch(error=>{
    console.warn('Showcase results unavailable:',error);
    preview.textContent='Case scores unavailable';
  });
})();


/* Subgroup performance matrix */
(async function initSubgroupLeaderboard() {
  const root = document.querySelector('[data-subgroup]');
  if (!root) return;
  const dimSelect = root.querySelector('[data-subgroup-dimension]');
  const metricSelect = root.querySelector('[data-subgroup-metric]');
  const table = root.querySelector('[data-subgroup-table]');
  const headings = root.querySelector('[data-subgroup-headings]');
  const colgroup = root.querySelector('[data-subgroup-cols]');
  const groupHeader = root.querySelector('[data-subgroup-group]');
  const groupSubtitle = root.querySelector('[data-subgroup-table-title]');
  const status = root.querySelector('[data-subgroup-order]');
  const note = root.querySelector('[data-subgroup-note]');
  const tbody = table.tBodies[0];
  const directions = {dice:'desc', nsd:'desc', under:'asc', over:'asc', speed:'asc'};
  const metricLabels = {dice:'Dice', nsd:'NSD', under:'Under', over:'Over', speed:'Speed'};
  const dimensionLabels = {organ:'Organ',stain:'Stain',center:'Center',source:'Dataset source'};
  const format = (v,k) => !Number.isFinite(v) ? '—' :
    k==='under'||k==='over' ? (v*100).toFixed(1)+'%' : v.toFixed(k==='speed'?2:3);
  const sortValue = v => Number.isFinite(v) ? v : null;
  let manifest, overall;
  try {
    [manifest, overall] = await Promise.all([
      fetch('assets/benchmark-subgroups.json', {cache:'no-store'}).then(r=>{if(!r.ok) throw Error(`Subgroups HTTP ${r.status}`);return r.json()}),
      fetch('assets/benchmark-results.json', {cache:'no-store'}).then(r=>{if(!r.ok) throw Error(`Benchmark HTTP ${r.status}`);return r.json()})
    ]);
    if(manifest.schema_version !== 1 || manifest.cohort_key !== overall.cohort_key ||
       manifest.matched_slides !== overall.matched_slides || !Array.isArray(overall.methods))
      throw Error('Subgroups and overall leaderboard were generated from different cohorts');
  } catch(err) {
    console.error('Subgroup leaderboard unavailable:',err);
    dimSelect.disabled=true;metricSelect.disabled=true;
    groupSubtitle.textContent='Results unavailable';
    note.textContent='Subgroup data unavailable. Rerun query_showcase.py to regenerate the benchmark exports.';
    return;
  }

  const state = {dimension:'organ',metric:'dice',active:'overall',direction:'desc'};
  // Subgroup categories are derived from exported results.
  function getGroups() {
    const data=manifest.groupings?.[state.dimension]?.groups || [];
    return data.filter(g => g.n >= manifest.min_group_slides && g.scores && Object.keys(g.scores).length);
  }
  function score(row, key) {
    if(key==='overall') return sortValue(row.method[state.metric]);
    const g=row.groupByLabel.get(key);
    return g ? sortValue(g.scores[row.method.run_id]?.[state.metric]) : null;
  }
  function compare(a,b,key,direction) {
    const av=score(a,key),bv=score(b,key);
    if(av===null) return bv===null ? a.method.method.localeCompare(b.method.method):1;
    if(bv===null) return -1;
    return (direction==='asc'?av-bv:bv-av)||a.method.method.localeCompare(b.method.method);
  }
  function render() {
    const groups=getGroups();
    const labels=['overall',...groups.map(g=>g.label)];
    if(!labels.includes(state.active)) state.active='overall';
    const better=directions[state.metric];
    groupHeader.colSpan=labels.length;
    groupSubtitle.textContent=`${dimensionLabels[state.dimension]} · ${metricLabels[state.metric]}`;
    table.style.setProperty('--subgroup-columns',String(labels.length));
    colgroup.replaceChildren();
    const methodCol=document.createElement('col');methodCol.className='lb-col-method';colgroup.append(methodCol);
    headings.replaceChildren();
    labels.forEach(label=>{
      const metricCol=document.createElement('col');metricCol.className='lb-col-metric';colgroup.append(metricCol);
      const th=document.createElement('th');th.scope='col';
      th.dataset.subgroupColumn=label;
      if(state.active===label)th.setAttribute('aria-sort',state.direction==='asc'?'ascending':'descending');
      const button=document.createElement('button');button.type='button';
      button.dataset.subgroupSort=label;
      const pretty=label==='overall'?'Overall':label;
      button.setAttribute('aria-label',`Sort by ${pretty} ${metricLabels[state.metric]} (${better==='asc'?'lower':'higher'} is better)`);
      button.title=`${metricLabels[state.metric]} · ${better==='asc'?'lower':'higher'} is better`;
      // Keep subgroup headers concise.
      const name=document.createElement('span');name.className='subgroup-column-name';name.textContent=pretty;
      button.append(name);th.append(button);headings.append(th);
      button.addEventListener('click',()=>{
        state.direction=state.active===label ? (state.direction==='asc'?'desc':'asc') : better;
        state.active=label;
        render();
      });
    });
    const groupByLabel=new Map(groups.map(g=>[g.label,g]));
    const rows=overall.methods.map(method=>({method,groupByLabel}));
    const best=[...rows].sort((a,b)=>compare(a,b,state.active,better));
    const cap=leaderboardDisplayLimit(overall);
    // Apply the display limit after ranking by the selected subgroup.
    const selected=Number.isFinite(cap)
      ? best.filter(r=>score(r,state.active)!==null).slice(0,cap)
      : best;
    const rank=new Map(best.map((r,i)=>[r.method.run_id,i+1]));
    selected.sort((a,b)=>compare(a,b,state.active,state.direction));
    tbody.replaceChildren();
    for(const row of selected) {
      const tr=document.createElement('tr');
      const th=document.createElement('th');th.scope='row';
      const position=document.createElement('span');position.className='lb-rank';position.textContent=String(rank.get(row.method.run_id)).padStart(2,'0');
      const identity=document.createElement('span');identity.className='lb-method';
      const full=row.method.method||row.method.label||'';
      const name=row.method.method_name || full.split('/')[0];
      const arch=row.method.architecture || (full.includes('/')?full.slice(full.indexOf('/')+1):'');
      const methodName=document.createElement('span');methodName.className='lb-method-name';methodName.textContent=name;
      identity.append(methodName);
      if(arch){const archName=document.createElement('span');archName.className='lb-architecture';archName.textContent=' / '+arch;identity.append(archName);}
      th.append(position,identity);tr.append(th);
      labels.forEach(label=>{
        const cell=document.createElement('td');const value=score(row,label);
        cell.textContent=format(value,state.metric);
        if(value===null) cell.classList.add('lb-unavailable');
        if(label===state.active) cell.classList.add('lb-active-value');
        tr.append(cell);
      });
      tbody.append(tr);
    }
    status.textContent=`${Number.isFinite(cap)?`Top ${cap}`:'All methods'} · ${state.active==='overall'?'Overall':state.active} ${metricLabels[state.metric]} · ${state.direction===better?'best first':'reverse order'}`;
    note.textContent='Each cell is the mean score for the selected subgroup. Click any column to rank methods; overall scores provide a reference.';
  }
  dimSelect.addEventListener('change',()=>{state.dimension=dimSelect.value;state.active='overall';state.direction=directions[state.metric];render();});
  metricSelect.addEventListener('change',()=>{state.metric=metricSelect.value;state.active='overall';state.direction=directions[state.metric];render();});
  render();
})();


/* Speed–accuracy plot with Pareto-efficient methods */
(async function initTradeoff() {
  const root = document.querySelector('[data-tradeoff]');
  if (!root) return;
  const metricControl = root.querySelector('[data-tradeoff-metric]');
  const frontierControl = root.querySelector('[data-tradeoff-frontier]');
  const svg = root.querySelector('[data-tradeoff-svg]');
  const detail = root.querySelector('[data-tradeoff-detail]');
  const empty = root.querySelector('[data-tradeoff-empty]');
  const NS='http://www.w3.org/2000/svg';
  const create = (tag,attrs={},value) => {
    const e = document.createElementNS(NS,tag);
    Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));
    if(value!==undefined) e.textContent=String(value);
    return e;
  };
  let data;
  try {
    const r = await fetch('assets/benchmark-results.json',{cache:'no-store'});
    if(!r.ok) throw Error(`Benchmark HTTP ${r.status}`);
    data = await r.json();
  } catch(err) {
    console.warn('Speed-accuracy data unavailable:',err);
    detail.textContent='Benchmark data unavailable';
    svg.hidden=true;empty.hidden=false;empty.textContent='Could not load benchmark results.';
    return;
  }
  const entries=(Array.isArray(data.methods)?data.methods:[]).filter(m=>
    m && typeof m.run_id==='string' && m.cohort_key===data.cohort_key &&
    m.matched_slides===data.matched_slides &&
    Number.isFinite(m.speed) && m.speed>0 &&
    Number.isFinite(m.dice) && m.dice>=0 && m.dice<=1 &&
    Number.isFinite(m.nsd) && m.nsd>=0 && m.nsd<=1);
  if(!entries.length) {
    svg.hidden=true;empty.hidden=false;detail.textContent='No timed methods to compare';
    return;
  }
  let selected=null;
  function pareto(methods,metric) {
    // Dominance requires strict improvement in at least one dimension.
    return methods.filter(a=>!methods.some(b=>b!==a &&
      b.speed<=a.speed && b[metric]>=a[metric] &&
      (b.speed<a.speed || b[metric]>a[metric])));
  }
  function render() {
    const metric=metricControl.value==='nsd'?'nsd':'dice';
    svg.replaceChildren();
    const W=1040,H=440,L=84,R=23,T=27,B=68;
    const cw=W-L-R,ch=H-T-B;
    const speeds=entries.map(m=>m.speed);
    const observedMin=Math.min(...speeds),observedMax=Math.max(...speeds);
    // Use 1–2–5 logarithmic tick intervals.
    const niceFloor=v=>{const power=10**Math.floor(Math.log10(v));return [5,2,1].find(a=>a*power<=v)*power;};
    const niceCeil=v=>{const power=10**Math.floor(Math.log10(v));return ([1,2,5,10].find(a=>a*power>=v)||10)*power;};
    const mn=Math.max(.001,niceFloor(observedMin));
    const mx=niceCeil(observedMax);
    // Handle coincident timing values.
    const lo=Math.log10(mn),hi=Math.log10(mx===mn?mx*2:mx);
    const x=s=>L+(Math.log10(s)-lo)/(hi-lo)*cw;
    const y=s=>T+(1-s)*ch;
    for(let step=0;step<=4;step++) {
      const val=step/4,py=y(val);
      svg.append(create('line',{x1:L,x2:W-R,y1:py,y2:py,class:'tradeoff-tick-line'}));
      svg.append(create('text',{x:L-13,y:py+4,'text-anchor':'end'},val.toFixed(2).replace(/0$/,'').replace(/\.$/,'')));
    }
    const ticks=[];
    for(let power=-3;power<=5;power++) for(const base of [1,2,5]) {
      const n=base*10**power;
      if(n>=mn && n<=mx)ticks.push(n);
    }
    for(const v of ticks) {
      const px=x(v);
      if(px<L-1||px>W-R+1)continue;
      svg.append(create('line',{x1:px,x2:px,y1:T,y2:H-B,class:'tradeoff-tick-line'}));
      svg.append(create('text',{x:px,y:H-B+25,'text-anchor':'middle'},v<1?v.toFixed(v<.1?2:1):String(v)));
    }
    svg.append(create('line',{x1:L,x2:L,y1:T,y2:H-B,class:'tradeoff-axis-line'}));
    svg.append(create('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'tradeoff-axis-line'}));
    // Align the metric label with the vertical axis.
    const axisCenter=(T+H-B)/2;
    svg.append(create('text',{
      x:18,y:axisCenter,class:'tradeoff-axis-title tradeoff-y-title',
      'text-anchor':'middle',transform:`rotate(-90 18 ${axisCenter})`
    },`Mean ${metric==='dice'?'Dice':'NSD'}`));
    // Keep the improvement arrow upright.
    svg.append(create('text',{x:L-1,y:T-12,class:'tradeoff-axis-direction','text-anchor':'middle'},'↑'));
    svg.append(create('text',{x:L+cw/2,y:H-15,class:'tradeoff-axis-title','text-anchor':'middle'},'Mean processing time (seconds / slide, log scale) →'));
    const efficient=pareto(entries,metric).sort((a,b)=>a.speed-b.speed||b[metric]-a[metric]);
    const frontierIds=new Set(efficient.map(m=>m.run_id));
    if(frontierControl.checked && efficient.length>1) {
      svg.append(create('polyline',{points:efficient.map(m=>`${x(m.speed)},${y(m[metric])}`).join(' '),class:'tradeoff-frontier-path'}));
    }
    // Retain the selected method when the metric changes.
    if(!selected || !entries.some(e=>e.run_id===selected)) selected=[...entries].sort((a,b)=>b[metric]-a[metric]||a.speed-b.speed)[0].run_id;
    function showDetail(m) {
      const label=m.method||m.label||m.run_id;
      const name=document.createElement('strong');name.textContent=label;
      const sub=document.createElement('span');
      sub.textContent=`  ·  ${metric.toUpperCase()} ${m[metric].toFixed(3)}  ·  ${m.speed.toFixed(2)} s/slide` +
        (frontierControl.checked && frontierIds.has(m.run_id)?'  ·  Pareto-efficient':'');
      detail.replaceChildren(name,sub);
    }
    // Render high-performing points last, followed by the selected method.
    const ordered=[...entries].sort((a,b)=>Number(frontierIds.has(a.run_id))-Number(frontierIds.has(b.run_id)));
    const drawPoint=m=>{
      const isFront=frontierControl.checked && frontierIds.has(m.run_id);
      const isSelected=m.run_id===selected;
      const r=isSelected?8:(isFront?6:5.1);
      const c=create('circle',{
        cx:x(m.speed),cy:y(m[metric]),r,
        class:'tradeoff-dot'+(isFront?' is-frontier':'')+(isSelected?' is-selected':''),
        tabindex:0,role:'button',
        'aria-label':`${m.method||m.label||m.run_id}; ${metric.toUpperCase()} ${m[metric].toFixed(3)}; ${m.speed.toFixed(2)} seconds per slide`
      });
      const tip=create('title',{},`${m.method||m.label||m.run_id}\n${metric.toUpperCase()} ${m[metric].toFixed(3)} · ${m.speed.toFixed(2)} s/slide`);
      c.append(tip);
      c.addEventListener('mouseenter',()=>showDetail(m));
      c.addEventListener('focus',()=>showDetail(m));
      c.addEventListener('mouseleave',()=>{const s=entries.find(v=>v.run_id===selected);if(s)showDetail(s);});
      c.addEventListener('blur',()=>{const s=entries.find(v=>v.run_id===selected);if(s)showDetail(s);});
      c.addEventListener('click',()=>{selected=m.run_id;render();});
      c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selected=m.run_id;render();}});
      svg.append(c);
    };
    ordered.filter(m=>m.run_id!==selected).forEach(drawPoint);
    const current=entries.find(m=>m.run_id===selected);
    drawPoint(current);
    showDetail(current);
    svg.setAttribute('aria-label',`Speed versus ${metric.toUpperCase()} for ${entries.length} measured methods. Lower processing time and higher accuracy are better. Select a point to inspect values.`);
  }
  metricControl.addEventListener('change',()=>{selected=null;render();});
  frontierControl.addEventListener('change',render);
  render();
})();
