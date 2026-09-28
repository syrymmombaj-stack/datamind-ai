const $ = id => document.getElementById(id);
const state = {sampleId:'wine',csv:null,session:null,samples:[]};
const esc = value => String(value ?? '—').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const fmt = value => typeof value === 'number' ? (Number.isInteger(value) ? String(value) : value.toFixed(3)) : String(value ?? '—');

async function request(path, body){
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : undefined);
  const data = await response.json();
  if(!response.ok) throw new Error(data.error || 'The request failed.');
  return data;
}
function status(message, kind=''){$('status').textContent=message;$('status').className='status '+kind;}
function payload(){return state.csv !== null ? {csv:state.csv} : {sample_id:state.sampleId};}
function renderTable(rows, columns, highlighted=false){
  if(!rows.length) return '<div class="placeholder">No rows available.</div>';
  return '<table><thead><tr>'+columns.map(c=>'<th>'+esc(c)+'</th>').join('')+'</tr></thead><tbody>'+rows.map((row,i)=>'<tr class="'+(highlighted&&i===0?'winner-row':'')+'">'+columns.map(c=>'<td>'+esc(fmt(row[c]))+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
}
function resetResults(){state.session=null;$('results').hidden=true;$('prediction-output').replaceChildren();}
async function inspect(){
  resetResults();$('train').disabled=true;
  status('Reading dataset…','busy');
  try{
    const data=await request('/api/inspect',payload());
    $('rows').textContent=data.rows.toLocaleString();$('features').textContent=data.columns.length-1;$('missing').textContent=data.missing.toLocaleString();
    $('target').innerHTML=data.columns.map(c=>'<option value="'+esc(c)+'">'+esc(c)+'</option>').join('');
    $('target').value=data.default_target;
    $('task').value=data.default_task;
    $('preview').innerHTML=renderTable(data.preview,data.columns);
    const sample=state.samples.find(s=>s.id===state.sampleId);
    $('dataset-name').textContent=state.csv!==null ? 'Your CSV' : sample?.name || 'Sample';
    $('train').disabled=false;status('Dataset ready. Choose a target, then train your models.');
  }catch(error){status(error.message,'error');$('preview').innerHTML='<div class="placeholder">Could not preview this dataset.</div>';}
}
function renderSamples(){
  $('sample-list').innerHTML=state.samples.map(s=>'<button class="sample-option '+(state.sampleId===s.id&&state.csv===null?'active':'')+'" data-sample="'+esc(s.id)+'"><span><strong>'+esc(s.name)+'</strong><small>'+esc(s.task)+' · '+(s.id==='wine'?'178':s.id==='iris'?'150':'442')+' rows</small></span><span class="arrow">↗</span></button>').join('');
  $('sample-list').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{state.sampleId=button.dataset.sample;state.csv=null;$('dataset-file').value='';$('file-name').textContent='';renderSamples();inspect();}));
}
function renderResults(data){
  $('results').hidden=false;$('winner').textContent=data.winner+' selected';$('split').textContent=data.training_rows+' train / '+data.test_rows+' test';
  const scoreColumns=Object.keys(data.scores[0]);$('scores').innerHTML=renderTable(data.scores,scoreColumns,true);
  const top=Math.max(.001,...data.importance.map(x=>Math.max(0,x.Importance)));
  $('importance').innerHTML=data.importance.map(x=>'<div class="bar-item"><div class="bar-caption"><span>'+esc(x.Feature)+'</span><span>'+esc(fmt(x.Importance))+'</span></div><div class="bar-track"><i style="width:'+Math.max(0,Math.min(100,x.Importance/top*100))+'%"></i></div></div>').join('')||'<p class="muted">No feature results.</p>';
  $('comparison').innerHTML=renderTable(data.comparison,['actual','predicted']);
  $('results').scrollIntoView({behavior:'smooth',block:'start'});
}

async function init(){
  try{
    state.samples=(await request('/api/samples')).samples;
    renderSamples();await inspect();
  }catch(error){status('The server is starting. Refresh this page in a moment.','error');}
}
$('dataset-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  if(file.size>10_000_000){status('CSV exceeds 10 MB.','error');return;}
  state.csv=await file.text();$('file-name').textContent=file.name;renderSamples();inspect();
});
$('train').addEventListener('click',async()=>{
  resetResults();$('train').disabled=true;status('Training three models with cross validation. This can take a few seconds…','busy');
  try{
    const data=await request('/api/train',{...payload(),target:$('target').value,task:$('task').value});
    state.session=data.session_id;renderResults(data);status('Training complete. Models evaluated on separate test rows.');
  }catch(error){status(error.message,'error');}
  finally{$('train').disabled=false;}
});
$('predict-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file||!state.session)return;
  if(file.size>1_000_000){status('Prediction CSV exceeds 1 MB.','error');return;}
  status('Running predictions…','busy');
  try{
    const data=await request('/api/predict',{session_id:state.session,csv:await file.text()});
    $('prediction-output').innerHTML='<p class="muted">'+data.rows+' rows predicted.</p>'+renderTable(data.preview,Object.keys(data.preview[0]));
    const blob=new Blob([data.csv],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);
    const link=document.createElement('a');link.href=url;link.download='datamind-predictions.csv';link.className='button primary';link.textContent='Download predictions ↓';$('prediction-output').appendChild(link);
    link.addEventListener('click',()=>setTimeout(()=>URL.revokeObjectURL(url),5000),{once:true});
    status('Predictions ready to download.');
  }catch(error){status(error.message,'error');}
  event.target.value='';
});
init();
