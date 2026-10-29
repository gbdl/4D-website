/* SPDX-License-Identifier: GPL-3.0-or-later */
// Small rendering control with a recording canvas, not scientific evidence.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('../assets/app.js',import.meta.url),'utf8').replace(/start\(\);\s*$/,'');
const records=[];
const ctx=new Proxy({},{set(o,k,v){o[k]=v;return true;},get(o,k){return k in o?o[k]:(...args)=>records.push({method:k,args,fill:o.fillStyle,stroke:o.strokeStyle});}});
const canvas={width:400,height:100,getContext:()=>ctx};
const scope={document:{getElementById:()=>null},window:{addEventListener:()=>{}},canvas,records};
vm.createContext(scope);
vm.runInContext(source+`\ndrawMap(canvas,{E:[[null,null,0,.2]],mask:[[false,true,true,true]],status:[[0,2,1,1]]},10,true);`,scope);
const cells=records.filter(r=>r.method==='fillRect').slice(1);
assert.equal(cells.length,4);
assert.equal(new Set(cells.map(c=>c.fill)).size,4,'outside, invalid, valid zero, saturation are distinct');
assert.equal(records.filter(r=>r.method==='stroke'&&r.stroke==='#a32945').length,1,'invalid cross');
assert.equal(records.filter(r=>r.method==='stroke'&&r.stroke==='#f6f3ec').length,1,'saturation hatch');
console.log(JSON.stringify({status:'passed',scope:'Synthetic rendering control only; no scientific arrays changed.',distinct_states:['outside','invalid','valid zero','above range'],invalid_cross:true,saturation_hatch:true},null,2));
