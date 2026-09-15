#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology_runtime import ExecutionAwareTopology

ROOT=Path(__file__).resolve().parents[1]
def main():
    g=json.loads((ROOT/'outputs/formal/RelationNav/topology/spatial_semantic_topology.json').read_text())
    s='interior_0135_840032:'
    t=ExecutionAwareTopology(g)
    source=s+'room_01'; target=s+'room_03'
    before=t.route(source,target); assert before and len(before)>=2
    first_via=before[0]['via']; t.observe(first_via,'blocked')
    after=t.route(source,target); assert after and first_via not in [e['via'] for e in after]
    print(json.dumps({'source':source,'target':target,'initial_route':[e['via'] for e in before], 'blocked_edge':first_via, 'rerouted':[e['via'] for e in after]}))
if __name__=='__main__': main()
