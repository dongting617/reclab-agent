"""Frozen functional coding trials with equal limits and hidden independent checks."""
from __future__ import annotations
import difflib
import json
import math
from pathlib import Path
import time
from reclab_agent.agents import create_agent
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.contracts import canonical_json,content_hash,identifier
from reclab_agent.experiments.workspace import Workspace
from reclab_agent.infrastructure.files import atomic_write,confined_path
from reclab_agent.tools.coding import CodingToolbox
from reclab_agent.evaluation.costs import summarize_usage

PROMPT='Read src/solution.py, implement solve(case) for the specification, validate syntax and use the public container checks. Only this source file is editable. Return a concise conclusion; hidden cases and reference solutions are unavailable. '


def check_rows(executor,workspace,rows):
    execution=executor.run(workspace,[r['input'] for r in rows])
    outputs=execution.get('outputs')
    def equal(a,b):
        if isinstance(a,list) and isinstance(b,list):
            return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        if type(a) in (int,float) and type(b) in (int,float):
            return math.isfinite(a) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
        return type(a) is type(b) and a==b
    passed=sum(equal(a,row['expected']) for a,row in zip(outputs,rows)) if outputs is not None and not execution['error'] else 0
    return {'passed':passed,'total':len(rows),'error':execution['error'],
            'exit_code':execution['exit_code'],'elapsed_ms':execution['elapsed_ms'],'stderr':execution.get('stderr','')}


def run_coding_suite(manifest_path,output,*,executor,mode='reference',backend=None,
                     repetitions=2,max_steps=8,max_requests=64):
    path=Path(manifest_path).absolute();output=Path(output).absolute()
    manifest=read_json_strict(path,max_bytes=1_000_000)
    frozen=read_json_strict(path.with_name('coding-freeze.json'),max_bytes=16000)
    digest=content_hash(manifest)
    if frozen['manifest']!=path.name or frozen['sha256']!=digest:
        raise ValueError('coding manifest changed after freezing')
    if mode not in {'reference','react','team'} or type(repetitions) is not int or not 1<=repetitions<=2:
        raise ValueError('coding evaluation supports reference/react/team and one or two trials')
    if type(max_steps) is not int or not 1<=max_steps<=16 or type(max_requests) is not int or max_requests<1:
        raise ValueError('coding request limits must be positive and bounded')
    identity=content_hash({'manifest':digest,'mode':mode,'model':getattr(backend,'model',None),
        'image':executor.image,'repetitions':repetitions,'max_steps':max_steps,'max_requests':max_requests,
        'prompt':PROMPT,'policy':'completed_changed_source_and_hidden_checks_v1'})
    report={'kind':'isolated_functional_coding','identity':identity,'manifest_hash':digest,'mode':mode,
        'model':getattr(backend,'model',None),'image':executor.image,'gpu_jobs':0,'status':'running',
        'scope':manifest['origin'],'limits':{'model_requests_per_trial':max_steps,'patches_per_trial':4,'public_checks_per_trial':3},
        'model_requests':0,'results':[],'limitations':['Small correlated synthetic coding cases are not production repository repair or recommendation quality.','Used requests and tokens may differ despite identical upper limits.']}
    if output.exists():
        report=read_json_strict(output,max_bytes=4_000_000)
        if report['identity']!=identity:raise ValueError('coding output belongs to a different frozen evaluation')
    def save():atomic_write(output,canonical_json(report).encode())
    trace_root=output.parent/(output.stem+'-traces')
    active=report.pop('active_case',None)
    if active and not any(r['id']==active['id'] for r in report['results']):
        report['results'].append({'id':active['id'],'status':'interrupted','passed':False,'error_type':'PriorTrialInterrupted'})
    save()
    done={r['id'] for r in report['results']}
    for case in manifest['cases']:
        identifier(case['id'])
        for rep in range(1,repetitions+1):
            trial=case['id']+'-'+str(rep)
            if trial in done:continue
            row={'id':trial,'case':case['id'],'repetition':rep,'status':'failed','passed':False}
            if report['model_requests']>=max_requests:
                row.update(status='not_run',error_type='RequestLimit');report['results'].append(row);save();continue
            started=time.monotonic();before=report['model_requests'];trial_calls=0
            report['active_case']={'id':trial};save()
            root=confined_path(trace_root,trial+'/workspace')
            workspace=Workspace(root,['src/solution.py'])
            atomic_write(root/'src/solution.py',case['starter'].encode())
            tools=CodingToolbox(workspace,lambda:check_rows(executor,root,case['public']))
            trace=confined_path(trace_root,trial+'/trace.jsonl')
            class LimitedBackend:
                model=getattr(backend,'model',None)
                def respond(self,**kwargs):
                    nonlocal trial_calls
                    if report['model_requests']>=max_requests or trial_calls>=max_steps:
                        raise ValueError('coding model request allowance exhausted')
                    trial_calls+=1;report['model_requests']+=1;save()
                    return backend.respond(**kwargs)
            try:
                if mode=='reference':
                    atomic_write(root/'src/solution.py',case['reference'].encode())
                    agent_status='reference_recipe'
                else:
                    result=create_agent(mode,tools,LimitedBackend(),max_steps=max_steps,model_retries=0,trace_path=trace).run(
                        PROMPT+case['description'],task_id=trial,allowed_tools={s.name for s in tools.specs})
                    agent_status=result['status']
                final=workspace.read('src/solution.py')['text']
                public=check_rows(executor,root,case['public']);private=check_rows(executor,root,case['private'])
                row.update(status='completed',agent_status=agent_status,public=public,private=private,
                    source_hash=content_hash(final),changed_source=final!=case['starter'],
                    diff=''.join(difflib.unified_diff(case['starter'].splitlines(True),final.splitlines(True),fromfile='before.py',tofile='after.py')),
                    passed=agent_status in {'completed','reference_recipe'} and final!=case['starter']
                        and public['passed']==public['total'] and private['passed']==private['total'])
            except Exception as error:
                row['error_type']=type(error).__name__
            row.update(model_requests=report['model_requests']-before,patch_attempts=tools.patches,
                public_check_calls=tools.checks,elapsed_ms=round((time.monotonic()-started)*1000,3))
            report['results'].append(row);report.pop('active_case',None);save()
    from reclab_agent.evaluation.traces import load_trace
    events=[]
    for trace in trace_root.glob('*/trace.jsonl'):events.extend(load_trace(trace))
    report.update(status='completed' if all(r['status']=='completed' for r in report['results']) else 'partial',
        trials=len(report['results']),passed=sum(r['passed'] for r in report['results']),resource_usage=summarize_usage(events))
    report['pass_rate']=report['passed']/report['trials'] if report['trials'] else 0
    usage=report['resource_usage']
    missing=max(0,report['model_requests']-usage['model_requests'])
    usage.update(reserved_attempts=report['model_requests'],requests_without_usage=usage['requests_without_usage']+missing,
                 usage_complete=usage['usage_complete'] and missing==0)
    save();return report
