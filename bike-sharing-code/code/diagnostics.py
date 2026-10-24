"""Coverage, matched day/night comparison, scale and cleaning sensitivity."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from common import OUT,CACHE,TEXT,SEED,clock,dump

def main():
    t=pd.concat([pd.read_parquet(CACHE/f'trips-{y}.parquet') for y in [2023,2024]],ignore_index=True)
    c=pd.read_parquet(CACHE/'counters.parquet');a=pd.read_csv(OUT/'counter-regimes.csv')
    f=pd.read_parquet(CACHE/'site-features.parquet')
    availability=a.assign(year=pd.to_datetime(a.date).dt.year).groupby(['site','year']).agg(complete_regimes=('date','size'),days=('date','nunique'))
    availability.to_csv(OUT/'site-coverage.csv')
    compare=[]
    for site in ['Ercolani','Parri','Sabotino','San Donato']:
        z=a.loc[a.site==site].merge(f[['site','date','regime','global_trips']],on=['site','date','regime'])
        complete=z.groupby('date').regime.nunique();z=z.loc[z.date.isin(complete.loc[complete==2].index)]
        z['all_counts']=z.inward+z.outward
        for year in [2023,2024]:
            v=z.loc[z.date.str.startswith(str(year))]
            compare.append({'site':site,'year':year,'days':v.date.nunique(),
                'counter_night_share':v.loc[v.regime=='night','all_counts'].sum()/v.all_counts.sum(),
                'sharing_night_share':v.loc[v.regime=='night','global_trips'].sum()/v.global_trips.sum(),
                'counter_total':v.all_counts.sum(),'sharing_total':v.global_trips.sum()})
    comparison=pd.DataFrame(compare);comparison.to_csv(OUT/'matched-night-shares.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.1,2.5));names=[]
    for year,marker in [(2023,'o'),(2024,'s')]:
        z=comparison.loc[comparison.year==year]
        ax.scatter(z.site,z.counter_night_share*100,marker=marker,label=f'Counters {year}',s=35)
        ax.scatter(z.site,z.sharing_night_share*100,marker=marker,facecolors='none',edgecolors='black',label=f'Sharing {year}',s=35)
    ax.set(ylabel='Night share (%)',title='Matched complete service days; sharing is citywide, counters are local')
    ax.legend(ncol=2,fontsize=7);fig.tight_layout();fig.savefig(TEXT/'figures/night-comparison.pdf',bbox_inches='tight');plt.close(fig)
    # Finer/rougher cells: geographic, not a traffic assignment model.
    out=[]
    for km in [.5,1,2]:
        cols={prefix+axis:np.floor(t[prefix+axis]/km).astype('int16') for prefix in ['o','d'] for axis in ['x','y']}
        z=pd.DataFrame(cols);z['year']=t.year;z['regime']=t.regime
        p={k:v.groupby(['ox','oy','dx','dy']).size() for k,v in z.groupby(['year','regime'])}
        p=pd.concat(p,axis=1).fillna(0)
        for year in [2023,2024]:
            x=p[(year,'day')].to_numpy();y=p[(year,'night')].to_numpy()
            rng=np.random.default_rng(SEED)
            equal=[]
            for _ in range(30):
                xx=rng.multinomial(100000,x/x.sum());yy=rng.multinomial(100000,y/y.sum())
                equal.append(jensenshannon(xx,yy,base=2))
            out.append({'km':km,'year':year,'js_day_night':float(jensenshannon(x,y,base=2)),
                        'equal_n_js_mean':float(np.mean(equal)),'equal_n_js_sd':float(np.std(equal))})
    pd.DataFrame(out).to_csv(OUT/'od-scale-sensitivity.csv',index=False)
    sensitivity=[]
    for min_min,max_min,speed in [(0,0,0),(.5,180,45),(1,180,45),(2,180,45),(.5,60,45),(.5,180,30),(2,60,30)]:
        keep=pd.Series(True,index=t.index) if speed==0 else t.duration.between(min_min,max_min)&(t.chord/t.duration*60<=speed)
        for year in [2023,2024]:
            z=t.loc[keep&(t.year==year)]
            sensitivity.append({'min_min':min_min,'max_min':max_min,'max_chord_speed':speed,'year':year,'n':len(z),
                'night_share':(z.regime=='night').mean(),'electric_share':(z.type=='electric').mean()})
    pd.DataFrame(sensitivity).to_csv(OUT/'cleaning-sensitivity.csv',index=False)
    # Supplementary 1 km day/night persistence: each endpoint uses its own timestamp.
    o=t.loc[t.start.dt.year==2024].groupby(['date','regime','ogx','ogy']).size().rename('departures')
    o.index.names=['date','regime','gx','gy']
    returns=t.loc[t.end.dt.year==2024,['end','dgx','dgy']].copy()
    returns['date'],returns['regime']=clock(returns.end)
    d=returns.groupby(['date','regime','dgx','dgy']).size().rename('arrivals')
    d.index.names=['date','regime','gx','gy']
    b=pd.concat([o,d],axis=1).fillna(0);b['balance']=b.arrivals-b.departures
    b=b.loc[b.arrivals+b.departures>=10].reset_index()
    p=b.groupby(['regime','gx','gy']).agg(active_days=('date','nunique'),sink_fraction=('balance',lambda s:(s>0).mean()),source_fraction=('balance',lambda s:(s<0).mean()),mean_balance=('balance','mean'))
    p.to_csv(OUT/'balance-persistence.csv')
    # No names/geocoding are inferred for spatial cells.
    dump('diagnostic-design.json',{'service_day_boundary':6,'night_start':23,'night_end_exclusive':6,
          'balance_map_cell_km':.5,'balance_map_blocks_hours':[[0,6],[6,12],[12,18],[18,24]],
          'balance_map_min_activity':200,'persistence_cell_km':1,'persistence_daily_min_activity':10,
          'arrival_assignment':'Recorded return time, not departure time.',
         'od_scale_km':[.5,1,2],'equal_size_multinomial_n':100000,'equal_size_replicates':30,
          'cleaning_sensitivity':'All populations already exclude recorded durations below 30 s; zero triple means no additional duration/speed screen.'})

if __name__=='__main__':main()
