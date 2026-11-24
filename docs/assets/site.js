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




const valid = value => typeof value === 'number' && Number.isFinite(value);
const subsets = ['twi', 'cpg_rumc'];
const subsetNames = {twi: 'Stains-in-the-wild', cpg_rumc: 'CPG-RUMC'};
const metrics = ['dice', 'under', 'over', 'seconds'];
function publicLabel(record) {
  return record.method;
}
async function initLeaderboard() {
  const table = document.querySelector('[data-leaderboard]');
  if (!table) return;
  const status = document.querySelector('[data-order-status]');
  const updated = document.querySelector('[data-benchmark-updated]');
  const body = table.tBodies[0];
  let data;
  try {
    const response = await fetch('assets/benchmark-results.json', {cache:'no-store'});
    if (!response.ok) throw Error(`Benchmark HTTP ${response.status}`);
    data = await response.json();
    if (!data.datasets || !data.cohorts || !subsets.every(key => Array.isArray(data.datasets[key]))) throw Error('Regenerate benchmark results');
  } catch (error) {
    status.textContent = 'Benchmark results unavailable.';
    document.querySelectorAll('[data-tradeoff]').forEach(root => {
      root.querySelector('[data-tradeoff-detail]').textContent = 'Benchmark results unavailable.';
      root.querySelector('[data-tradeoff-empty]').hidden = false;
      root.querySelector('[data-tradeoff-svg]').hidden = true;
    });
    return;
  }
  if (data.generated_at && Number.isFinite(Date.parse(data.generated_at))) {
    updated.dateTime = new Date(data.generated_at).toISOString();
    updated.textContent = `Last updated ${new Intl.DateTimeFormat('en-GB',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'UTC'}).format(new Date(data.generated_at))} UTC`;
    updated.hidden = false;
  }
  const grouped = new Map();
  subsets.forEach(key => {
    data.datasets[key].forEach(record => {
      if (record.status !== 'in_progress' && (record.rank == null || record.ranked_n !== data.cohorts[key])) return;
      if (!grouped.has(record.id)) grouped.set(record.id, {identity:record});
      grouped.get(record.id)[key] = record;
    });
  });
  const expand = document.querySelector('[data-leaderboard-expand]');
  const view = document.querySelector('[data-leaderboard-view]');
  const topIds = Object.fromEntries(subsets.map(key => {
    const seen = new Set();
    const ranked = data.datasets[key].filter(record => record.status === 'complete' && record.rank != null && record.ranked_n === data.cohorts[key] && valid(record.dice)).sort((a,b) => a.rank-b.rank);
    const best = ranked.filter(record => {
      if (seen.has(record.method)) return false;
      seen.add(record.method);
      return true;
    }).slice(0,10);
    return [key,new Set(best.map(record => record.id))];
  }));
  let sortKey = 'twi:dice', direction = 'desc', expanded = false;
  function render() {
    const [subset, metric] = sortKey.split(':');
    const rows = [...grouped.values()].filter(entry => expanded || topIds[subset].has(entry.identity.id));
    rows.sort((a,b) => {
      const av=a[subset]?.[metric], bv=b[subset]?.[metric];
      if (valid(av) && valid(bv) && av!==bv) return direction==='desc'?bv-av:av-bv;
      if (valid(av)!==valid(bv)) return valid(av)?-1:1;
      return (a[subset]?.rank ?? Infinity)-(b[subset]?.rank ?? Infinity) || (a.twi?.rank ?? Infinity)-(b.twi?.rank ?? Infinity) || (a.cpg_rumc?.rank ?? Infinity)-(b.cpg_rumc?.rank ?? Infinity) || a.identity.id-b.identity.id;
    });
    body.replaceChildren();
    rows.forEach(entry => {
      const record=entry.identity;
      const row=document.createElement('tr');
      const heading=document.createElement('th'); heading.scope='row';
      const method=document.createElement('span'); method.className='lb-method';
      const name=document.createElement('span'); name.className='lb-method-name'; name.textContent=record.method;
      method.append(name);
      const secondary=expanded ? [record.mpp==null?'':`${record.mpp} MPP`,record.configuration].filter(Boolean).join(' · ') : '';
      if (secondary) {
        const architecture=document.createElement('span'); architecture.className='lb-architecture'; architecture.textContent=secondary; method.append(architecture);
      }
      heading.append(method); row.append(heading);
      subsets.forEach(key => {
        if (entry[key]?.status === 'in_progress') {
          const cell=document.createElement('td');
          cell.colSpan=4; cell.className='lb-progress'; cell.textContent='Evaluation in progress';
          row.append(cell); return;
        }
        metrics.forEach(metric => {
        const cell=document.createElement('td'); const value=entry[key]?.[metric];
        if (!valid(value)) { cell.textContent='-'; cell.className='lb-unavailable'; }
        else if (metric==='under'||metric==='over') cell.textContent=(value*100).toFixed(1)+'%';
        else cell.textContent=value.toFixed(metric==='dice'?3:2);
        if (valid(value) && `${key}:${metric}`===sortKey) cell.classList.add('lb-active-value');
        row.append(cell);
        });
      });
      body.append(row);
    });
    table.querySelectorAll('[data-sort-key]').forEach(th => {
      th.removeAttribute('aria-sort');
      if (th.dataset.sortKey===sortKey) th.setAttribute('aria-sort',direction==='desc'?'descending':'ascending');
    });
    status.textContent=rows.length?`Sorted by ${subsetNames[subset]} / ${metric==='seconds'?'s / slide':metric==='dice'?'Dice':metric}`:'No fully evaluated configurations available.';
    if (view) view.textContent = expanded ? `${rows.length} configurations` : `Top ${rows.length} methods by Dice · ${subsetNames[subset]}`;
    if (expand) {
      expand.hidden = grouped.size <= topIds[subset].size;
      expand.setAttribute('aria-expanded',String(expanded));
      expand.replaceChildren(document.createTextNode(expanded?'Show top 10 methods ':`Show all ${grouped.size} configurations `));
      const arrow = document.createElement('span');
      arrow.setAttribute('aria-hidden','true');
      arrow.textContent = expanded?'↑':'↓';
      expand.append(arrow);
    }
  }
  expand?.addEventListener('click',()=>{ expanded = !expanded; render(); });
  table.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{
    const key=button.dataset.sort;
    direction=key===sortKey?(direction==='asc'?'desc':'asc'):(key.endsWith(':dice')?'desc':'asc');
    sortKey=key; render();
  }));
  render();
  document.querySelectorAll('[data-tradeoff]').forEach(root=>initTradeoff(data,root));
}
initLeaderboard();
function initTradeoff(data,root) {
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
  const subset=root.dataset.subset;
  const entries=(data.datasets[subset] || []).filter(m => m.rank != null && m.ranked_n===data.cohorts[subset] && m.pareto_rank != null && m.pareto_rank <= 10 && Number.isFinite(m.pareto_seconds) && m.pareto_seconds > 0 && Number.isFinite(m.pareto_dice) && m.pareto_n > 0).map(m => ({...m, dice:m.pareto_dice, speed:m.pareto_seconds, run_id:String(m.id), method:publicLabel(m)}));
  root.querySelector('.tradeoff-description').textContent = entries.length ? `${entries.length} methods · ${entries[0].pareto_n} matched slides` : 'No matched timing data';
  svg.hidden = !entries.length;
  empty.hidden = !!entries.length;
  if (!entries.length) { svg.replaceChildren(); detail.textContent='No timed methods to compare'; return; }
  let selected=null;
  function pareto(methods,metric) {

    return methods.filter(a=>!methods.some(b=>b!==a &&
      b.speed<=a.speed && b[metric]>=a[metric] &&
      (b.speed<a.speed || b[metric]>a[metric])));
  }
  function render() {
    if (!entries.length) return;
    const metric='dice';
    svg.replaceChildren();
    const W=640,H=440,L=62,R=28,T=30,B=68;
    svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
    const cw=W-L-R,ch=H-T-B;
    const speeds=entries.map(m=>m.speed);
    const observedMin=Math.min(...speeds),observedMax=Math.max(...speeds);

    const niceFloor=v=>{const power=10**Math.floor(Math.log10(v));return [5,2,1].find(a=>a*power<=v)*power;};
    const niceCeil=v=>{const power=10**Math.floor(Math.log10(v));return ([1,2,5,10].find(a=>a*power>=v)||10)*power;};
    const mn=niceFloor(observedMin*.75);
    const mx=niceCeil(observedMax*1.3);

    const lo=Math.log10(mn),hi=Math.log10(mx===mn?mx*2:mx);
    const x=s=>L+(Math.log10(s)-lo)/(hi-lo)*cw;
    const low=Math.max(0,Math.floor((Math.min(...entries.map(m=>m.dice))-.04)*20)/20);
    const high=Math.min(1,Math.ceil((Math.max(...entries.map(m=>m.dice))+.04)*20)/20);
    const y=s=>T+(high-s)/(high-low)*ch;
    for(let step=0;step<=4;step++) {
      const val=low+(high-low)*step/4,py=y(val);
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

    const axisCenter=(T+H-B)/2;
    svg.append(create('text',{
      x:18,y:axisCenter,class:'tradeoff-axis-title tradeoff-y-title',
      'text-anchor':'middle',transform:`rotate(-90 18 ${axisCenter})`
    },`Mean ${metric==='dice'?'Dice':'NSD'}`));

    svg.append(create('text',{x:L-1,y:T-12,class:'tradeoff-axis-direction','text-anchor':'middle'},'↑'));
    svg.append(create('text',{x:L+cw/2,y:H-15,class:'tradeoff-axis-title','text-anchor':'middle'},'Mean processing time (seconds / slide, log scale) →'));
    const efficient=pareto(entries,metric).sort((a,b)=>a.speed-b.speed||b[metric]-a[metric]);
    const frontierIds=new Set(efficient.map(m=>m.run_id));
    if(frontierControl.checked && efficient.length>1) {
      svg.append(create('polyline',{points:efficient.map(m=>`${x(m.speed)},${y(m[metric])}`).join(' '),class:'tradeoff-frontier-path'}));
    }

    if(!selected || !entries.some(e=>e.run_id===selected)) selected=[...entries].sort((a,b)=>b[metric]-a[metric]||a.speed-b.speed)[0].run_id;
    function showDetail(m) {
      const label=m.method||m.label||m.run_id;
      const name=document.createElement('strong');name.textContent=label;
      const sub=document.createElement('span');
      sub.textContent=`  ·  ${metric.toUpperCase()} ${m[metric].toFixed(3)}  ·  ${m.speed.toFixed(2)} s/slide` +
        (frontierControl.checked && frontierIds.has(m.run_id)?'  ·  Pareto-efficient':'');
      detail.replaceChildren(name,sub);
    }

    const ordered=[...entries].sort((a,b)=>Number(frontierIds.has(a.run_id))-Number(frontierIds.has(b.run_id)));
    const drawPoint=m=>{
      const isFront=frontierControl.checked && frontierIds.has(m.run_id);
      const isSelected=m.run_id===selected;
      const r=isSelected?8:(isFront?6:5.1);
      const c=create('circle',{
        cx:x(m.speed),cy:y(m[metric]),r,
        class:'tradeoff-dot'+(isFront?' is-frontier':'')+(isSelected?' is-selected':''),
        tabindex:0,role:'button',
        'aria-label':`${m.method||m.label||m.run_id}, ${metric.toUpperCase()} ${m[metric].toFixed(3)}, ${m.speed.toFixed(2)} seconds per slide`
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
    const occupied=[];
    const points=entries.map(m=>({x:x(m.speed),y:y(m[metric])}));
    const overlaps=(a,b)=>a.x < b.x+b.width && a.x+a.width > b.x && a.y < b.y+b.height && a.y+a.height > b.y;
    const markers=points.map(p=>({x:p.x-10,y:p.y-10,width:20,height:20}));
    const labelled=[...entries].sort((a,b)=>b.method.length-a.method.length || b.dice-a.dice);
    labelled.forEach(m=>{
      const px=x(m.speed),py=y(m[metric]);
      const cls=frontierControl.checked && frontierIds.has(m.run_id)?' is-frontier':'';
      const text=create('text',{x:0,y:0,class:'tradeoff-node-label'+cls},m.method);
      svg.append(text);
      const bounds=text.getBBox();
      const options=[];
      for(const d of [10,16,24,36,50,70,95,125]) {
        options.push([d,-9,'start'],[-d,-9,'end'],[d,15,'start'],[-d,15,'end'],[0,-d,'middle'],[0,d+5,'middle']);
        if(d>16) options.push([d,-d,'start'],[-d,-d,'end'],[d,d,'start'],[-d,d,'end']);
      }
      let chosen=null;
      for(const [dx,dy,anchor] of options) {
        const lx=px+dx,ly=py+dy;
        const box={x:lx+bounds.x-(anchor==='middle'?bounds.width/2:anchor==='end'?bounds.width:0)-2,y:ly+bounds.y-2,width:bounds.width+4,height:bounds.height+4};
        if(box.x<L || box.x+box.width>W-R || box.y<T || box.y+box.height>H-B || [...occupied,...markers].some(b=>overlaps(box,b))) continue;
        chosen={lx,ly,anchor,box,dx,dy}; break;
      }
      if(!chosen) {
        for(let ly=T+15;ly<H-B-3&&!chosen;ly+=15) {
          for(let lx=L+5;lx<W-R-bounds.width-5;lx+=20) {
            const box={x:lx-2,y:ly+bounds.y-2,width:bounds.width+4,height:bounds.height+4};
            if(![...occupied,...markers].some(b=>overlaps(box,b))) {chosen={lx,ly,anchor:'start',box,dx:lx-px,dy:ly-py};break;}
          }
        }
      }
      if(!chosen) {text.remove();return;}
      const {lx,ly,anchor,box,dx,dy}=chosen;
      occupied.push(box);
      text.setAttribute('x',lx); text.setAttribute('y',ly); text.setAttribute('text-anchor',anchor);
      if(Math.hypot(dx,dy)>28) {
        const ex=Math.max(box.x,Math.min(px,box.x+box.width)),ey=Math.max(box.y,Math.min(py,box.y+box.height));
        const length=Math.hypot(ex-px,ey-py)||1;
        svg.insertBefore(create('line',{x1:px+7*(ex-px)/length,y1:py+7*(ey-py)/length,x2:ex,y2:ey,class:'tradeoff-label-line'}),svg.querySelector('.tradeoff-dot'));
      }
    });
    showDetail(current);
    svg.setAttribute('aria-label',`Speed versus ${metric.toUpperCase()} for ${entries.length} measured methods. Lower processing time and higher accuracy are better. Select a point to inspect values.`);
  }
  frontierControl.addEventListener('change',render);
  render();
}
