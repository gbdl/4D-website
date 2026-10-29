/* GPL-3.0. Static inspection of frozen scientific samples; no metric evaluation. */
const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
const COLORS={raw022:'#286d78',quotient087:'#c2410c',Raw:'#286d78',Periodic:'#7d6196',Quotient:'#c2410c'};
const NAMES={raw022:'Local metric',quotient087:'Quotient metric'};
const IDS=['raw022','quotient087'];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const fmt=(x,n=4)=>!finite(x)?'—':x!==0&&Math.abs(x)<.0001?x.toExponential(2):Number(x.toPrecision(n)).toString();
const seq=(a,b,n=5)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
const median=a=>{const b=[...a].sort((x,y)=>x-y);const i=Math.floor(b.length/2);return b.length%2?b[i]:(b[i-1]+b[i])/2;};
let metrics,experiments,audit,selectedCell={row:16,col:16},selectedRun='22';
function el(tag,attrs={},text){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
function add(p,tag,a={},t){const n=el(tag,a,t);p.append(n);return n;}
function makeChart(id,opts={}){
 const svg=$(id);svg.replaceChildren();const vb=svg.viewBox.baseVal;
 const base=svg._baseViewBox||(svg._baseViewBox={width:vb.width,height:vb.height});
 const narrow=window.innerWidth<700,W=narrow?Math.max(280,svg.getBoundingClientRect().width):base.width,H=narrow?Math.max(290,Math.min(350,base.height)):base.height;
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const margin={left:66,right:20,top:30,bottom:52,...opts.margin,...(narrow?{left:id==='paired-chart'?88:55,right:14,top:45,bottom:57}:{})};const box={x:margin.left,y:margin.top,w:W-margin.left-margin.right,h:H-margin.top-margin.bottom};
 let xd=opts.xdomain||[0,1],yd=opts.ydomain||[0,1];if(xd[0]===xd[1])xd=[xd[0],xd[0]+1];if(yd[0]===yd[1])yd=[yd[0],yd[0]+1];
 const tr=(v,log)=>log?Math.log10(v):v;
 const x=v=>box.x+(tr(v,opts.xlog)-tr(xd[0],opts.xlog))/(tr(xd[1],opts.xlog)-tr(xd[0],opts.xlog))*box.w;
 const y=v=>box.y+box.h-(tr(v,opts.ylog)-tr(yd[0],opts.ylog))/(tr(yd[1],opts.ylog)-tr(yd[0],opts.ylog))*box.h;
 const ticks=(d,log)=>{if(!log)return seq(...d);let out=[];for(let p=Math.ceil(Math.log10(d[0]));p<=Math.floor(Math.log10(d[1]));p++)out.push(10**p);return out.length>1?out:seq(...d);};
 for(const v of opts.xticks||ticks(xd,opts.xlog)){add(svg,'line',{x1:x(v),x2:x(v),y1:box.y,y2:box.y+box.h,class:'chart-grid'});add(svg,'text',{x:x(v),y:box.y+box.h+22,'text-anchor':'middle',class:'chart-text'},fmt(v,3));}
 for(const v of opts.yticks||ticks(yd,opts.ylog)){add(svg,'line',{x1:box.x,x2:box.x+box.w,y1:y(v),y2:y(v),class:'chart-grid'});add(svg,'text',{x:box.x-10,y:y(v)+4,'text-anchor':'end',class:'chart-text'},fmt(v,3));}
 add(svg,'line',{x1:box.x,y1:box.y+box.h,x2:box.x+box.w,y2:box.y+box.h,class:'chart-axis'});
 function label(text,x,y,anchor,width){
  const words=text.split(' '),lines=[''],limit=Math.max(15,Math.floor(width/6.3));
  for(const word of words){const i=lines.length-1;if(narrow&&lines[i].length+word.length+1>limit)lines.push(word);else lines[i]+=(lines[i]?' ':'')+word;}
  const node=add(svg,'text',{x,y,'text-anchor':anchor,class:'chart-label'});
  lines.forEach((line,i)=>add(node,'tspan',{x,dy:i?14:0},line));
 }
 label(opts.xlabel||'',box.x+box.w/2,H-(narrow?24:8),'middle',W-35);
 label(opts.ylabel||'',narrow?8:box.x,15,'start',W-20);
 const clipId=id+'-clip';add(add(svg,'defs'),'clipPath',{id:clipId}).append(el('rect',{x:box.x,y:box.y,width:box.w,height:box.h}));
 const layer=add(svg,'g',{'clip-path':`url(#${clipId})`});
 function line(values,color,width=2,dash=''){let d='',open=false;for(const p of values){if(!p||!finite(p[0])||!finite(p[1])||(opts.xlog&&p[0]<=0)||(opts.ylog&&p[1]<=0)){open=false;continue;}d+=(open?'L':'M')+x(p[0]).toFixed(3)+','+y(p[1]).toFixed(3);open=true;}add(layer,'path',{d,fill:'none',stroke:color,'stroke-width':width,'stroke-dasharray':dash});}
 function dot(xv,yv,color,title,r=3,click){if(!finite(xv)||!finite(yv)||(opts.ylog&&yv<=0))return;const n=add(layer,'circle',{cx:x(xv),cy:y(yv),r,fill:color,opacity:.85,...(click?{tabindex:0,role:'button','aria-label':title,class:'chart-dot'}:{})});add(n,'title',{},title);if(click){n.addEventListener('click',click);n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();click();}});}return n;}
 function reference(v,label){if(v<yd[0]||v>yd[1])return;add(layer,'line',{x1:box.x,x2:box.x+box.w,y1:y(v),y2:y(v),stroke:'#7c8075','stroke-dasharray':'5 4'});if(label)add(svg,'text',{x:box.x+box.w-4,y:y(v)-6,'text-anchor':'end',class:'chart-text'},label);}
 return {svg,layer,box,x,y,line,dot,reference,xd,yd};
}
const palette=[[241,242,215],[169,203,177],[91,155,155],[57,95,134],[53,38,78]];
function heatColor(f){const t=Math.max(0,Math.min(1,f))*4,i=Math.min(3,Math.floor(t)),u=t-i;return `rgb(${palette[i].map((v,k)=>Math.round(v+(palette[i+1][k]-v)*u)).join(',')})`;}
function drawMap(canvas,map,cap,mini=false,labels={x:'x',y:'y'}){
 const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);ctx.fillStyle='#ffffff';ctx.fillRect(0,0,W,H);
 const b=mini?{x:0,y:0,w:W,h:H}:{x:62,y:18,w:W-85,h:H-72};const rows=map.E.length,cols=map.E[0].length,cw=b.w/cols,ch=b.h/rows;
 for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
  const x=b.x+col*cw,y=b.y+(rows-1-row)*ch,v=map.E[row][col],inside=map.mask[row][col];
  ctx.fillStyle=!inside?'#e1e2db':!finite(v)||map.status[row][col]!==1?'#f4dce3':heatColor(100*v/cap);ctx.fillRect(x,y,cw+.3,ch+.3);
  if(!inside){ctx.strokeStyle='#c7c9c2';ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(x,y+ch);ctx.lineTo(x+cw,y);ctx.stroke();}
  else if(!finite(v)||map.status[row][col]!==1){ctx.strokeStyle='#a32945';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x+2,y+2);ctx.lineTo(x+cw-2,y+ch-2);ctx.moveTo(x+2,y+ch-2);ctx.lineTo(x+cw-2,y+2);ctx.stroke();}
  else if(100*v>cap){ctx.strokeStyle='#f6f3ec';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(x,y+ch);ctx.lineTo(x+cw,y);ctx.stroke();}
 }
 if(!mini){ctx.strokeStyle='#9da396';ctx.lineWidth=1;ctx.strokeRect(b.x,b.y,b.w,b.h);ctx.fillStyle='#626760';ctx.font=(canvas.clientWidth<420?'19':'13')+'px system-ui';ctx.textAlign='center';
  for(const i of [0,8,16,24,32]){const col=Math.min(cols-1,i),row=Math.min(rows-1,i);ctx.fillText(fmt(map.axis0[col],3),b.x+(col+.5)*cw,b.y+b.h+23);ctx.textAlign='right';ctx.fillText(fmt(map.axis1[row],3),b.x-10,b.y+(rows-row-.5)*ch+4);ctx.textAlign='center';}
  ctx.font=(canvas.clientWidth<420?'20':'14')+'px system-ui';ctx.fillText(labels.x,b.x+b.w/2,H-8);ctx.save();ctx.translate(17,b.y+b.h/2);ctx.rotate(-Math.PI/2);ctx.fillText(labels.y,0,0);ctx.restore();
  const x=b.x+(selectedCell.col+.5)*cw,y=b.y+(rows-selectedCell.row-.5)*ch;ctx.strokeStyle='#232c2b';ctx.lineWidth=1;ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(x,b.y);ctx.lineTo(x,b.y+b.h);ctx.moveTo(b.x,y);ctx.lineTo(b.x+b.w,y);ctx.stroke();ctx.setLineDash([]);
 }
 canvas._plotBox={...b,rows,cols};
}
function geometry(){
 const z=Number($('height').value),rj=2.80701761247; $('height-value').textContent=fmt(z,3);
 const chart=makeChart('domain-chart',{xdomain:[-1,3],ydomain:[-1,3],xlabel:'x₂',ylabel:'x₃',xticks:[-1,0,1,2,3],yticks:[-1,0,1,2,3]});
 const fold=x=>x<0?-x:x<=1?x:x<=2?2-x:x-2,C=[[1,0,0],[0,1,0],[0,0,1],[1,1,1]];let d='';const n=88,step=4/n;
 for(let j=0;j<n;j++)for(let i=0;i<n;i++){let x=-1+(i+.5)*step,y=-1+(j+.5)*step;const u=[.5,fold(x),fold(y)];if(C.every(c=>z*z+u.reduce((a,v,k)=>a+(v-c[k])**2,0)>=1)){const xx=chart.x(x-step/2),yy=chart.y(y+step/2);d+=`M${xx.toFixed(2)},${yy.toFixed(2)}h${(chart.box.w/n+.2).toFixed(2)}v${(chart.box.h/n+.2).toFixed(2)}h-${(chart.box.w/n+.2).toFixed(2)}Z`;}}
 add(chart.layer,'path',{d,fill:'#bad4cf'});
 for(const v of [0,1,2]){add(chart.layer,'line',{x1:chart.x(v),x2:chart.x(v),y1:chart.y(-1),y2:chart.y(3),stroke:'#7b9e96','stroke-dasharray':'3 5'});add(chart.layer,'line',{x1:chart.x(-1),x2:chart.x(3),y1:chart.y(v),y2:chart.y(v),stroke:'#7b9e96','stroke-dasharray':'3 5'});}
 const svg=$('core-chart');svg.replaceChildren();const cx=155,cy=122,R=89,rr=R*Math.sqrt(Math.max(0,1-z/rj));
 add(svg,'circle',{cx,cy,r:R,fill:'#eef0e6',stroke:'#d0d6c9','stroke-dasharray':'4 5'});add(svg,'line',{x1:cx-R-15,x2:cx+R+15,y1:cy,y2:cy,stroke:'#bdc6b8'});add(svg,'line',{y1:cy-R-15,y2:cy+R+15,x1:cx,x2:cx,stroke:'#bdc6b8'});add(svg,'circle',{cx,cy,r:Math.max(rr,.5),fill:'none',stroke:'#286d78','stroke-width':2});add(svg,'line',{x1:cx,y1:cy,x2:cx+rr,y2:cy,stroke:'#c2410c','stroke-width':2});add(svg,'circle',{cx:cx+rr,cy,r:4,fill:'#c2410c'});add(svg,'circle',{cx,cy,r:2,fill:'#252c2a'});add(svg,'text',{x:cx+R+22,y:cy+5,class:'chart-label'},'U');add(svg,'text',{x:cx+4,y:cy-R-18,class:'chart-label'},'V');add(svg,'text',{x:cx,y:246,'text-anchor':'middle',class:'chart-label'},`ρ = ${fmt(Math.sqrt(4*rj*(rj-z)/3),4)}`);
 $('geometry-caption').textContent=z<1?'Below z = 1, the hemisphere inequalities cut away part of each section.':z<rj-1e-6?'Above z = 1, the full horizontal box is present. The regular radial coordinate continues toward zero.':'At z = rⱼ, use the regular core chart. The ordinary-coordinate metric expression is singular.';
}
function mapStatus(m,row,col){return !m.mask[row][col]?'outside domain':m.status[row][col]!==1||!finite(m.E[row][col])?'invalid':'valid';}
function maps(rebuildTable=true){
 const region=$('region').value,cap=Number($('map-scale').value),label=metrics.map_labels[region];
 $('scale-label').textContent=cap+'%';
 for(const[id,cid]of [['raw022','map-local'],['quotient087','map-compatible']]){const m=metrics.states[id].maps[region];drawMap($(cid),m,cap,false,label);$(cid+'-caption').textContent=`${metrics.states[id].id}, step ${metrics.states[id].step.toLocaleString()}, ${m.mask.flat().filter(Boolean).length} admissible samples`;}
 $('map-coordinate').textContent=`${label.chart} chart, ${label.fixed}. The same saved coordinates and mask are used for both states.`;
 const a=metrics.states.raw022.maps[region],b=metrics.states.quotient087.maps[region],r=selectedCell.row,c=selectedCell.col;
 const vals=[a,b].map(m=>mapStatus(m,r,c)==='valid'?fmt(100*m.E[r][c],5)+'%':mapStatus(m,r,c));
 const coordinates=(a.coords||a.ordinary)?.[r]?.[c];
 $('map-inspect').textContent=`${label.x} = ${fmt(a.axis0[c],5)}, ${label.y} = ${fmt(a.axis1[r],5)}, Local metric: ${vals[0]}, Quotient metric: ${vals[1]}${coordinates?', '+(a.coords?'native':'ordinary')+' coordinates ('+coordinates.map(v=>fmt(v,5)).join(', ')+')':''}`;
 if(rebuildTable){$('map-table').querySelector('tbody').innerHTML=a.E.flatMap((row,r)=>row.map((v,c)=>`<tr><td>${fmt(a.axis0[c],5)}</td><td>${fmt(a.axis1[r],5)}</td><td>${mapStatus(a,r,c)==='valid'?fmt(100*v,5):'—'}</td><td>${mapStatus(b,r,c)==='valid'?fmt(100*b.E[r][c],5):'—'}</td><td>${esc(mapStatus(a,r,c))} / ${esc(mapStatus(b,r,c))}</td></tr>`)).join('');}
}
function mapInput(canvas){canvas.addEventListener('click',e=>{const r=canvas.getBoundingClientRect(),b=canvas._plotBox,x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height;if(x<b.x||x>b.x+b.w||y<b.y||y>b.y+b.h)return;selectedCell={col:Math.min(b.cols-1,Math.floor((x-b.x)/b.w*b.cols)),row:Math.min(b.rows-1,b.rows-1-Math.floor((y-b.y)/b.h*b.rows))};maps(false);});canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();selectedCell.col=Math.max(0,Math.min(32,selectedCell.col+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0)));selectedCell.row=Math.max(0,Math.min(32,selectedCell.row+(e.key==='ArrowUp'?1:e.key==='ArrowDown'?-1:0)));maps(false);});}
function profiles(){
 const key=$('profile').value,cusp=key.startsWith('cusp'),p=metrics.states.raw022.profiles[key],q=metrics.states.quotient087.profiles[key];$('resolution').disabled=cusp;
 const inds=!cusp&&$('resolution').value==='coarse'?p.coarse_indices:p.parameter.map((_,i)=>i), valid=(f,i)=>f.status[i]===1&&finite(f.E[i]);
 const points=f=>inds.map(i=>valid(f,i)?[f.parameter[i],100*f.E[i]]:null),spoints=f=>inds.map(i=>valid(f,i)&&finite(f.scalar[i])?[f.parameter[i],f.scalar[i]+12]:null);
 const all=[...points(p),...points(q)].filter(Boolean),alls=[...spoints(p),...spoints(q)].filter(Boolean),maxE=Math.max(.02,...all.map(v=>v[1]))*1.1,minS=Math.min(0,...alls.map(v=>v[1])),maxS=Math.max(0,...alls.map(v=>v[1])),pad=Math.max(.001,(maxS-minS)*.1);
 const xd=[p.parameter[0],p.parameter.at(-1)],xlabel=cusp?'Native height t (log scale)':'Ordinary height z',ce=makeChart('profile-e',{xdomain:xd,ydomain:[0,$('profile-scale').value==='zoom'?1:maxE],xlog:cusp,xlabel,ylabel:'Einstein residual, 100E (%)',...(cusp?{xticks:[1,2,4,8,16,32,64]}:{})}),cs=makeChart('profile-s',{xdomain:xd,ydomain:[minS-pad,maxS+pad],xlog:cusp,xlabel,ylabel:'Scalar curvature deviation, S + 12',...(cusp?{xticks:[1,2,4,8,16,32,64]}:{})});
 ce.reference(1,'1% reference');cs.reference(0,'Einstein value');ce.line(points(p),COLORS.raw022);ce.line(points(q),COLORS.quotient087);cs.line(spoints(p),COLORS.raw022);cs.line(spoints(q),COLORS.quotient087);
 const above=all.filter(v=>v[1]>ce.yd[1]).length;if(above){add(ce.svg,'text',{x:ce.box.x+8,y:ce.box.y+15,class:'chart-text'},`${above} samples above enlarged range ↑`);}
 const c=p.ordinary?.[0],fixed=c&&!cusp?`x = (${c.slice(1).map(v=>fmt(v,5)).join(', ')})`:cusp?`native site ${p.sites[0]}, transverse coordinates (${p.coords[0].slice(1).map(v=>fmt(v,5)).join(', ')})`:'';
 $('profile-caption').textContent=`${inds.length} saved samples per construction, ${fixed}. ${cusp?'Native chart; t = 1 through 64.':'The radial path ends at the filled core and is evaluated using the regular chart.'} ${key==='radial-quotient-transition'?'This line passes through an anomaly found in the saved cap-reflection panel; it need not contain the later audit’s largest error.':''}`;
 $('profile-table').querySelector('tbody').innerHTML=inds.map(i=>`<tr><td>${fmt(p.parameter[i],6)}</td><td>${valid(p,i)?fmt(100*p.E[i],6):'invalid'}</td><td>${valid(q,i)?fmt(100*q.E[i],6):'invalid'}</td><td>${finite(p.scalar[i])?fmt(p.scalar[i]+12,6):'invalid'}</td><td>${finite(q.scalar[i])?fmt(q.scalar[i]+12,6):'invalid'}</td></tr>`).join('');
}
function logDomain(a){const p=a.filter(v=>finite(v)&&v>0);return [10**Math.floor(Math.log10(Math.min(...p))),10**Math.ceil(Math.log10(Math.max(...p)))];}
function hypothesis(r){if(r.family==='Raw')return `Core-repair coefficient η = ${r.eta}; otherwise continued local Einstein and face training. Zero-penalty controls test continued improvement without the supplement.`;if(r.family==='Periodic')return `Periodic representation; ridge weight ${r.ridge}; ${r.pool?.startsWith('new')?'independently regenerated':'inherited'} large pool. This tests sampling and modest ridge repair.`;return `Quotient representation; ${r.sampling} sampling; conditional transition weight ${r.collar}. These starts are changed by projection, rather than preserving the founder metric.`;}
function experimentView(){
 const policy=$('checkpoint-policy').value,records=Object.values(experiments.records),states=records.map(r=>r.states[policy]);
 const chart=makeChart('experiment-chart',{xdomain:logDomain(states.map(s=>s.rms)),ydomain:logDomain(states.map(s=>s.maximum)),xlog:true,ylog:true,xlabel:'Worst regional RMS (%), log scale',ylabel:'Largest panel error (%), log scale'});
 chart.reference(1,'1% sampled-peak reference');
 for(const r of records){const s=r.states[policy],active=String(r.task??Number(r.id.slice(3)))===selectedRun;const d=chart.dot(s.rms,s.maximum,COLORS[r.family],`${r.id}: RMS ${fmt(s.rms)}%, maximum ${fmt(s.maximum)}%, step ${s.step}`,active?6:3.8,()=>{selectedRun=String(Number(r.id.slice(3)));$('experiment').value=selectedRun;experimentView();});if(active){d?.setAttribute('stroke','#252c2a');d?.setAttribute('stroke-width','2');}}
 const r=experiments.records[selectedRun],s=r.states[policy],parent=String(r.parent).replace(/task(\d+)-terminal\.npz/,'R3-$1 terminal');
 $('experiment-details').innerHTML=`<h4>${esc(r.id)} <span class="${r.family==='Raw'?'local':r.family==='Quotient'?'compatible':'periodic'}">/ ${esc(r.family)}</span></h4><p>${esc(hypothesis(r))}</p><dl><dt>Evaluated step</dt><dd>${s.step.toLocaleString()}</dd><dt>Worst RMS</dt><dd>${fmt(s.rms,5)}%</dd><dt>Panel maximum</dt><dd>${fmt(s.maximum,5)}%</dd><dt>Learning rate</dt><dd>${r.lr}</dd><dt>Parent</dt><dd>${esc(parent)}</dd><dt>Training</dt><dd>${fmt(r.training_hours,3)} h recorded</dd></dl><p class="small">${r.architecture.depth} × ${r.architecture.width}, ${esc(r.architecture.activation)}, selected-state policy: ${esc(policy)}.</p>`;
 const rows=experiments.curves[selectedRun].rows,xd=[0,Math.max(...rows.map(x=>x.hours))];
 const ce=makeChart('learning-e',{xdomain:xd,ydomain:logDomain(rows.map(x=>x.bulk_rms_percent)),ylog:true,xlabel:'Training time (hours)',ylabel:'Worst main-region RMS (%)'});ce.line(rows.map(x=>[x.hours,x.bulk_rms_percent]),COLORS[r.family]);
 const cf=makeChart('learning-f',{xdomain:xd,ydomain:logDomain(rows.flatMap(x=>[x.H_rms,x.B_rms])),ylog:true,xlabel:'Training time (hours)',ylabel:'Fixed-reference face RMS (log scale)'});cf.line(rows.map(x=>[x.hours,x.H_rms]),'#286d78');cf.line(rows.map(x=>[x.hours,x.B_rms]),'#c2410c',2,'4 3');add(cf.svg,'text',{x:cf.box.x+10,y:cf.box.y+15,class:'chart-text'},'blue: induced metric   orange dashed: bending');
 $('experiment-table').querySelector('tbody').innerHTML=records.map(r=>{const s=r.states[policy];return `<tr><td>${esc(r.id)}</td><td>${esc(r.family)}</td><td>${s.step}</td><td>${fmt(s.rms,6)}</td><td>${fmt(s.maximum,6)}</td></tr>`;}).join('');
}
const interventionLabels={'quotient_collar=1.0':'Quotient, transition emphasis 1','quotient_collar=0.25':'Quotient, transition emphasis 0.25','quotient_stratification':'Quotient, stratification alone','periodic_ridge':'Periodic, ridge pressure','periodic_newpool':'Periodic, new pool','raw_eta=0.01':'Raw, core penalty 0.01','raw_eta=0.1':'Raw, core penalty 0.1','raw_eta=1.0':'Raw, core penalty 1'};
function paired(){
 const label=$('intervention').value,rows=experiments.comparisons.filter(c=>c.label===label),values=rows.flatMap(x=>[x.Q_ratio,x.bulk_RMS_ratio]);
 const chart=makeChart('paired-chart',{xdomain:[Math.min(.25,...values)*.9,Math.max(2,...values)*1.05],ydomain:[-.6,rows.length-.4],xlog:true,xlabel:'Treatment / control, logarithmic scale',ylabel:'Matched runs: control → treatment',xticks:[.25,.5,1,2,4].filter(x=>x>=Math.min(.25,...values)*.9&&x<=Math.max(2,...values)*1.05),yticks:[],margin:{left:155,right:30}});
 add(chart.layer,'line',{x1:chart.x(1),x2:chart.x(1),y1:chart.box.y,y2:chart.box.y+chart.box.h,stroke:'#777f73','stroke-dasharray':'5 4'});
 rows.forEach((r,i)=>{const y=rows.length-1-i;add(chart.svg,'text',{x:chart.box.x-12,y:chart.y(y)+4,'text-anchor':'end',class:'chart-text'},`${String(r.control).padStart(3,'0')} → ${String(r.treatment).padStart(3,'0')}`);chart.line([[r.Q_ratio,y],[r.bulk_RMS_ratio,y]],'#b8bdae',1.5);chart.dot(r.Q_ratio,y,COLORS.raw022,`R4-${r.treatment}: Q ratio ${fmt(r.Q_ratio,5)}`,5);chart.dot(r.bulk_RMS_ratio,y,COLORS.quotient087,`R4-${r.treatment}: main-region RMS ratio ${fmt(r.bulk_RMS_ratio,5)}`,5);});
 const q=median(rows.map(r=>r.Q_ratio)),b=median(rows.map(r=>r.bulk_RMS_ratio));$('paired-caption').textContent=`Blue: common score Q. Orange: worst main-region RMS. ${rows.filter(r=>r.Q_ratio<1).length}/${rows.length} paired runs improved Q. Median ratios: Q ${fmt(q,4)}; main-region RMS ${fmt(b,4)}. Lower is better in each statistic.`;
}
function faces(){
 const quantity=$('face-quantity').value,summary=$('face-summary').value,key=quantity+'_'+summary,mult=quantity==='length'?100:1;
 const arrays=IDS.map(id=>metrics.states[id].physical_gluing.by_pair.map(p=>p[key]*mult)),chart=makeChart('face-chart',{xdomain:[0,95],ydomain:logDomain(arrays.flat()),ylog:true,xlabel:'External face-pair index',ylabel:quantity==='length'?'Directional length disagreement (%), log scale':'Bending (curvature-radius⁻¹), log scale',xticks:[0,16,32,48,64,80,95]});
 arrays.forEach((a,k)=>{for(let i=0;i<a.length;i++)chart.dot(i,a[i],COLORS[IDS[k]],`${NAMES[IDS[k]]}, pair ${i}: ${fmt(a[i],6)}`,2.8);});
 $('face-table').querySelector('tbody').innerHTML=arrays[0].map((v,i)=>`<tr><td>${i}</td><td>${fmt(v,7)}</td><td>${fmt(arrays[1][i],7)}</td></tr>`).join('');
}
function coreLinks(){
 const arrays=IDS.map(id=>metrics.states[id].core_links.signed_angle_defect.map(v=>Math.abs(v)*180/Math.PI)),chart=makeChart('core-links-chart',{xdomain:[0,351],ydomain:logDomain(arrays.flat().filter(v=>v>0)),ylog:true,xlabel:'Filled-core stratum index',ylabel:'Absolute closure-angle defect (degrees), log scale',xticks:[0,64,128,192,256,320,351]});
 arrays.forEach((a,k)=>a.forEach((v,i)=>{if(v>0)chart.dot(i,v,COLORS[IDS[k]],`${NAMES[IDS[k]]}, stratum ${i}: ${fmt(v,6)} degrees`,2);}));
 const zero=arrays.map(a=>a.filter(v=>v===0).length);$('core-links-caption').textContent=`Largest tested core defects: Local metric ${fmt(Math.max(...arrays[0]),4)}°; Quotient metric ${fmt(Math.max(...arrays[1]),4)}°. ${zero[0]+zero[1]?`${zero[0]} Local metric and ${zero[1]} Quotient metric values are exact floating-point zero and cannot appear on the logarithmic axis; they remain in the table.`:''} Ordinary-ridge measurements below use a separate catalogue.`;
 const a=metrics.states.raw022.core_links,b=metrics.states.quotient087.core_links;$('core-table').querySelector('tbody').innerHTML=a.stratum.map((s,i)=>`<tr><td>${esc(s)}</td><td>${esc(a.fibre_id[i])}</td><td>${fmt(a.signed_angle_defect[i]*180/Math.PI,7)}</td><td>${fmt(b.signed_angle_defect[i]*180/Math.PI,7)}</td></tr>`).join('');
}
function setHash(){const hash=new URLSearchParams({section:$('region').value,profile:$('profile').value,state:$('checkpoint-policy').value,run:selectedRun});history.replaceState(null,'','#view='+hash.toString());}
function loadHash(){if(!location.hash.startsWith('#view='))return;const p=new URLSearchParams(location.hash.slice(6));for(const[id,key]of [['region','section'],['profile','profile'],['checkpoint-policy','state']])if([...$(id).options].some(o=>o.value===p.get(key)))$(id).value=p.get(key);if(experiments.records[p.get('run')])selectedRun=p.get('run');$('experiment').value=selectedRun;}
function bind(id,fn){$(id).addEventListener('change',()=>{fn();setHash();});}
async function start(){
 geometry();$('height').addEventListener('input',geometry);
 try{
  const fetchData=async path=>{const r=await fetch(path);if(!r.ok)throw Error(`Could not load ${path}: ${r.status}`);return r.json();};
  [metrics,experiments,audit]=await Promise.all(['data/metrics.json','data/experiments.json','data/audit.json'].map(fetchData));
  $('experiment').innerHTML=Object.entries(experiments.records).sort((a,b)=>Number(a[0])-Number(b[0])).map(([k,r])=>`<option value="${k}">${esc(r.id)}, ${esc(r.family)}</option>`).join('');
  $('intervention').innerHTML=Object.entries(interventionLabels).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('');loadHash();$('experiment').value=selectedRun;
  drawMap($('hero-local'),metrics.states.raw022.maps.body,10,true);drawMap($('hero-compatible'),metrics.states.quotient087.maps.body,10,true);
  maps();profiles();experimentView();paired();faces();coreLinks();
  bind('region',()=>{selectedCell={row:16,col:16};maps();});bind('map-scale',maps);bind('profile',profiles);bind('resolution',profiles);bind('profile-scale',profiles);bind('checkpoint-policy',experimentView);bind('experiment',()=>{selectedRun=$('experiment').value;experimentView();});bind('intervention',paired);bind('face-quantity',faces);bind('face-summary',faces);mapInput($('map-local'));mapInput($('map-compatible'));
  $('load-status').classList.add('hidden');document.documentElement.dataset.ready='true';
 }catch(e){$('load-status').innerHTML=`The interactive evidence could not load. Reload the page; a downloaded copy must be opened through a local HTTP server, as described in the repository README. The <a href="assets/figures/matched-sections.png">saved figure</a> and <a href="data/tables/experiments.html">numerical tables</a> remain available.`;console.error(e);}
}
let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{geometry();if(metrics){maps(false);profiles();experimentView();paired();faces();coreLinks();}},120);});
start();
