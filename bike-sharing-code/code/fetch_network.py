"""Optional historical OSM acquisition. Not required by the reported experiment.

Successful retrieval is NOT validation of access rules, counter matching or routes.
Do not repeatedly rerun against a rate-limited public service.
"""
import argparse
import datetime
import json
import requests
from common import CACHE,OUT

QUERY='''[out:json][timeout:120][date:"2024-01-01T00:00:00Z"];
way["highway"]["highway"!~"motorway|motorway_link|construction|proposed|steps"]
(44.44,11.23,44.56,11.44);out body;>;out skel qt;'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',default='https://overpass-api.de/api/interpreter')
    args=parser.parse_args()
    record={'url':args.endpoint,'query':QUERY,'retrieved_utc':datetime.datetime.now(datetime.UTC).isoformat(),
            'license':'OpenStreetMap contributors, ODbL 1.0','validated_for_routing':False}
    try:
        response=requests.get(args.endpoint,params={'data':QUERY},timeout=150)
        record['http_status']=response.status_code;response.raise_for_status()
        data=response.json()
        if data.get('remark'):raise ValueError(data['remark'])
        if not data.get('elements'):raise ValueError('No network elements')
        (CACHE/'bologna-osm-2024.json').write_text(json.dumps(data))
        record['elements']=len(data['elements']);record['success']=True
    except Exception as exc:
        record['success']=False;record['error']=str(exc)
    (OUT/'network-acquisition-latest.json').write_text(json.dumps(record,indent=2))
    if not record['success']:raise SystemExit(record['error'])
    print('Downloaded raw OSM snapshot; independent validation still required.')

if __name__=='__main__':main()
