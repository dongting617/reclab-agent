"""Create the first frozen, small functional coding benchmark; never overwrite it."""
import hashlib
import json
from pathlib import Path
import random


def main():
    path=Path('tasks/coding.json');freeze=Path('tasks/coding-freeze.json')
    if path.exists() or freeze.exists():raise SystemExit('Coding benchmark already frozen; use a new version.')
    rng=random.Random(2026)
    cases=[]
    def add(name,description,starter,reference,inputs,answer):
        rows=[{'input':x,'expected':answer(x)} for x in inputs]
        cases.append({'id':name,'description':description,'starter':starter,'reference':reference,
                      'public':rows[:3],'private':rows[3:]})
    ranked=[{'scores':[1,2,2,0],'seen':[1],'k':3},{'scores':[],'seen':[],'k':3},
            {'scores':[1,3],'seen':[],'k':0}]
    ranked += [{'scores':[rng.randrange(5) for _ in range(n)],'seen':list(range(0,n,3)),
                'k':rng.randrange(0,n+3)} for n in range(1,25)]
    add('exclude_seen','Return up to k unseen zero-based item indices by descending score; break ties by index. k<=0 returns [].',
        'def solve(case):\n    return sorted(range(len(case["scores"])), key=lambda i: case["scores"][i], reverse=True)[:case["k"]]\n',
        'def solve(case):\n    seen = set(case["seen"])\n    return sorted((i for i in range(len(case["scores"])) if i not in seen), key=lambda i: (-case["scores"][i], i))[:max(0, case["k"])]\n',
        ranked,lambda x:sorted((i for i in range(len(x['scores'])) if i not in x['seen']),key=lambda i:(-x['scores'][i],i))[:max(0,x['k'])])
    means=[[0,2,4],[],[0,0]]+[[rng.randrange(-5,6) for _ in range(n)] for n in range(1,25)]
    add('masked_mean','Compute the arithmetic mean of nonzero values in the input list. Zero is padding; no nonzero values returns 0.0.',
        'def solve(case):\n    return sum(case) / len(case)\n',
        'def solve(case):\n    values = [v for v in case if v != 0]\n    return sum(values) / len(values) if values else 0.0\n',means,
        lambda x:sum(v for v in x if v!=0)/sum(v!=0 for v in x) if any(v!=0 for v in x) else 0.0)
    padded=[{'items':[1,2,3,4],'limit':2},{'items':[7],'limit':3},{'items':[],'limit':0}]
    padded += [{'items':list(range(1,n+1)),'limit':rng.randrange(0,10)} for n in range(1,25)]
    add('recent_history','Keep the most recent limit items and left-pad with zero to exactly limit values. A nonpositive limit returns [].',
        'def solve(case):\n    items = case["items"][:case["limit"]]\n    return items + [0] * (case["limit"] - len(items))\n',
        'def solve(case):\n    size = max(0, case["limit"])\n    tail = case["items"][-size:] if size else []\n    return [0] * (size - len(tail)) + tail\n',padded,
        lambda x:([0]*max(0,x['limit']-len(x['items']))+x['items'][-x['limit']:]) if x['limit']>0 else [])
    negatives=[{'num_items':4,'seen':[1],'target':3},{'num_items':0,'seen':[],'target':0},
               {'num_items':2,'seen':[1,2],'target':2}]
    negatives += [{'num_items':n,'seen':list(range(2,n+1,2)),'target':max(1,n//2)} for n in range(1,25)]
    add('negative_catalog','Return ascending item IDs from 1 through num_items, excluding seen IDs and the positive target. Never emit padding ID zero.',
        'def solve(case):\n    return sorted(set(range(case["num_items"])) - set(case["seen"]))\n',
        'def solve(case):\n    excluded = set(case["seen"]) | {case["target"]}\n    return [i for i in range(1, case["num_items"] + 1) if i not in excluded]\n',negatives,
        lambda x:[i for i in range(1,x['num_items']+1) if i not in x['seen'] and i!=x['target']])
    manifest={'schema_version':1,'origin':'Four synthetic functional recommendation-utility repairs; actual edits and isolated execution, not production research quality.',
              'seed':2026,'cases':cases}
    raw=json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    freeze.write_text(json.dumps({'manifest':'coding.json','sha256':hashlib.sha256(raw.encode()).hexdigest(),
                                 'policy':'Reference solutions and private expected values stay outside tool workspaces and containers.'},indent=2)+'\n',encoding='utf-8')
    print('Frozen four coding cases with 3 public and 24 private checks each.')


if __name__=='__main__':main()
