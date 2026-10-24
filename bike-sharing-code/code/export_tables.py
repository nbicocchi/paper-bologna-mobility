"""Generate manuscript numbers/tables directly from computed artifacts."""
import json
import platform
import hashlib
from importlib.metadata import version
import numpy as np
import pandas as pd
from common import ROOT,OUT,TEXT,dump

def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%')
def table(name,head,rows,align):
    tex='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(head)+r' \\'+'\n\\midrule\n'
    tex+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (TEXT/'tables'/f'{name}.tex').write_text(tex)
def main():
    audits=[json.loads((OUT/f'audit-sharing-{y}.json').read_text()) for y in [2023,2024]]
    descr=json.loads((OUT/'descriptive.json').read_text());syn=json.loads((OUT/'synthetic-design.json').read_text())
    counter=json.loads((OUT/'audit-counters.json').read_text())
    design=json.loads((OUT/'prediction-design.json').read_text())
    octo=pd.read_csv(OUT/'audit-monthly-2023.csv').set_index('month').loc[10]
    macro={'RawTrips':f"{sum(a['raw'] for a in audits):,}",'EndpointTrips':f"{sum(a['retained'] for a in audits):,}",
           'NightShare':f"{100*descr['night_share']:.1f}",'ElectricShare':f"{100*descr['electric_share']:.1f}",
           'OctRaw':f"{int(octo.unique_records):,}",'OctZero':f"{int(octo.zero_duration):,}",
           'OctZeroPct':f"{100*octo.zero_duration/octo.unique_records:.4f}",
           'OctRetained':f"{int(octo.retained):,}",'NullDifference':f"{100*syn['equivalent_od_relative_l1']:.1f}",
           'SyntheticRank':str(syn['rank_directional']),'SyntheticZeroColumns':str(syn['zero_columns'])}
    macro.update({'SelectedCounterSites':str(counter['sites']),'SelectedCounterDevices':str(counter['devices']),
        'CounterExcludedRows':f"{counter['excluded_rows']:,}",
        'CounterSelectedRawRows':f"{counter['input_rows']-counter['excluded_rows']:,}",
        'CounterMismatches':str(counter['total_mismatches'])})
    scores=pd.read_csv(OUT/'prediction-summary.csv').set_index(['split','model'])
    macro.update({'SpatialBasePooled':f"{100*scores.loc[('spatial_2023','calendar_weather'),'pooled_wape']:.1f}",
                   'SpatialChordPooled':f"{100*scores.loc[('spatial_2023','chord_proxy'),'pooled_wape']:.1f}",
                   'SpatialChordMacro':f"{100*scores.loc[('spatial_2023','chord_proxy'),'macro_wape']:.1f}",
                   'SpatialBaseMacro':f"{100*scores.loc[('spatial_2023','calendar_weather'),'macro_wape']:.1f}",
                   'SpatialUniformMacro':f"{100*scores.loc[('spatial_2023','uniform_chord'),'macro_wape']:.1f}",
                   'ForwardUniformPooled':f"{100*scores.loc[('spatiotemporal_2024','uniform_chord'),'pooled_wape']:.1f}",
                    'ForwardChordPooled':f"{100*scores.loc[('spatiotemporal_2024','chord_proxy'),'pooled_wape']:.1f}"})
    (TEXT/'tables/numbers.tex').write_text('% Generated; do not edit by hand.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macro.items()))
    table('audit',['Year','Raw rentals',r'Retained $\geq$30 s','Valid duration'],
          [[a['year'],f"{a['raw']:,}",f"{a['retained']:,}",f"{a['valid_duration_retained']:,}"] for a in audits],'rrrr')
    trip=pd.read_csv(OUT/'trip-summary.csv');rows=[]
    for r in trip.itertuples():
        rows.append([r.year,'E' if r.type=='electric' else 'M',r.regime,f'{r.n:,}',f'{r.distance_median:.2f}',f'{r.duration_median:.2f}'])
    table('trips',['Year','Type','Period','$n$','km$^a$','min$^b$'],rows,'rllrrr')
    cover=pd.read_csv(OUT/'site-coverage.csv');rows=[]
    for site,g in cover.groupby('site'):
        by=g.set_index('year').complete_regimes
        rows.append([esc(site),int(by.get(2023,0)),int(by.get(2024,0))])
    table('coverage',['Section group','2023 periods','2024 periods'],rows,'lrr')
    pred=pd.read_csv(OUT/'prediction-summary.csv')
    names={'calendar_weather':('Boosting','No sharing'),'global_sharing':('Boosting','Global sharing'),
        'local_endpoints':('Boosting','Local endpoints'),'chord_proxy':('Boosting','OD geometry'),
        'poisson_base':('Poisson','No sharing'),'poisson_global':('Poisson','Global sharing'),
        'poisson_local':('Poisson','Local endpoints'),'poisson_sharing':('Poisson','OD geometry'),
        'uniform_chord':('Scaling','250 m chord')}
    for split,label in [('spatial_2023','spatial'),('spatiotemporal_2024','forward')]:
        rows=[]
        for model,(family,features) in names.items():
            r=pred.loc[(pred.split==split)&(pred.model==model)].iloc[0]
            rows.append([family,features,f'{r.macro_wape*100:.1f}',f'{r.pooled_wape*100:.1f}',int(r.sites)])
        table('prediction-'+label,['Model','Features','Macro','Pooled','$S$'],rows,'llrrr')
    s=pd.read_csv(OUT/'synthetic-summary.csv');rows=[]
    for bias in [0,1]:
        for model in ['uniform_expansion','sharing_prior','gravity_prior']:
            r=s.loc[(s.bias_strength==bias)&(s.routing=='correct')&s.directional&(s.sections==10)&(s.model==model)].iloc[0]
            rows.append([bias,{'uniform_expansion':'Uniform expansion','sharing_prior':'Sharing prior','gravity_prior':'Gravity prior'}[model],
                         f'{r.test_wape_mean*100:.1f}',f'{r.od_relative_l1_mean*100:.1f}'])
    table('synthetic',['Bias','Estimator','Test WAPE','OD $L_1$'],rows,'rlrr')
    rows=[]
    for routing in ['correct','shortest_only']:
        for direct in [False,True]:
            r=s.loc[(s.bias_strength==1)&(s.routing==routing)&(s.directional==direct)&(s.sections==10)&(s.model=='sharing_prior')].iloc[0]
            rows.append(['Mixture' if routing=='correct' else 'Shortest','Dir.' if direct else 'Sum',f'{100*r.test_wape_mean:.1f}',f'{100*r.od_relative_l1_mean:.1f}'])
    table('route-ablation',['Assignment','Counts','Test WAPE','OD $L_1$'],rows,'llrr')
    js=pd.read_csv(OUT/'od-scale-sensitivity.csv');rows=[]
    for km in [.5,1,2]:
        v=js.loc[js.km==km].set_index('year');rows.append([km,f'{v.loc[2023,"js_day_night"]:.3f}',f'{v.loc[2024,"js_day_night"]:.3f}'])
    table('js',['Cell (km)','2023 day/night','2024 day/night'],rows,'rrr')
    reg=pd.read_csv(OUT/'regularization-sensitivity.csv');v=reg.loc[(reg.bias_strength==1)&(reg.model=='sharing_prior')].groupby('tau')[['test_wape','od_relative_l1']].mean()
    table('regularization',[r'$\tau$','Test WAPE','OD $L_1$'],[[tau,f'{r.test_wape*100:.1f}',f'{r.od_relative_l1*100:.1f}'] for tau,r in v.iterrows()],'rrr')
    versions={m:version(m) for m in ['numpy','pandas','scipy','matplotlib','scikit-learn','networkx','pyarrow','pyproj','requests','pillow']}
    dump('reproducibility.json',{'python':platform.python_version(),'packages':versions,
         'raw_files':{f"bike-sharing-{a['year']}.csv":a['sha256'] for a in audits},
         'counter_sha256':hashlib.file_digest((ROOT/'dataset/colonnine-conta-bici.csv').open('rb'),'sha256').hexdigest(),
         'weather_sha256':hashlib.file_digest((ROOT/'dataset/meteo-bologna-ilmeteo-2019-2025.csv').open('rb'),'sha256').hexdigest(),
         'code_sha256':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (ROOT/'code').glob('*.py')}})

if __name__=='__main__':main()
