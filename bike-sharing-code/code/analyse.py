"""Descriptive figures and privacy-preserving site/regime feature aggregates."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from common import ROOT,OUT,CACHE,TEXT,xy,dump
from citymap import draw_sharing_maps

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
COL={'electric':'#0072B2','mechanical':'#D55E00'}
def save(fig,name):
    fig.savefig(TEXT/'figures'/f'{name}.pdf');plt.close(fig)

def draw_trip_profiles(t):
    """Figures 1–3 use the map figures' font family, sizes, and design width."""
    m=t.groupby([t.start.dt.to_period('M'),'type']).size().unstack(fill_value=0)
    m.index=m.index.astype(str);m.to_csv(OUT/'monthly.csv')
    fig,ax=plt.subplots(figsize=(9,3))
    for kind in m:ax.plot(m.index,m[kind]/1000,label=kind,color=COL[kind],marker='.',lw=1.3)
    if '2023-10' in m.index:
        october=list(m.index).index('2023-10')
        ax.axvspan(october-.4,october+.4,color='gray',alpha=.18)
        ax.annotate('Timestamp defect:\nrecords removed by 30 s filter',
            xy=(october,0),xytext=(october+.5,m.to_numpy().max()/1000*.65),fontsize=8,
            arrowprops={'arrowstyle':'->','color':'gray'})
    ax.set(ylabel='Retained trips (thousands)',xlabel='Calendar month')
    ax.tick_params(axis='x',rotation=65);ax.legend(ncol=2);save(fig,'monthly')
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for ax,year in zip(axs,[2023,2024]):
        for wk,ls in [(False,'-'),(True,'--')]:
            for kind in COL:
                z=t.loc[(t.year==year)&(t.weekend==wk)&(t.type==kind)].hour.value_counts().sort_index().reindex(range(24),fill_value=0)
                ax.plot(z.index,z/z.sum()*100,color=COL[kind],ls=ls,label=f'{kind}, '+('weekend' if wk else 'weekday'))
        ax.axvspan(0,5.99,color='gray',alpha=.13);ax.axvspan(23,24,color='gray',alpha=.13)
        ax.set(title=str(year),xlabel='Local departure hour',ylabel='Within-group trips (%)',xlim=(0,23))
    axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'hourly')
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for kind in COL:
        for reg,ls in [('day','-'),('night','--')]:
            g=t.loc[(t.type==kind)&(t.regime==reg)]
            for ax,col,lim in zip(axs,['distance','duration'],[12,60]):
                a=np.sort(g[col].dropna().to_numpy());sel=np.linspace(0,len(a)-1,600).astype(int)
                ax.plot(a[sel],(sel+1)/len(a),color=COL[kind],ls=ls,label=f'{kind}, {reg}')
                ax.set(xlim=(0,lim),ylim=(0,1),xlabel='Recorded distance (km)' if col=='distance' else 'Duration (min)',ylabel='Empirical CDF')
    axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'distributions')

def main():
    ds=[]
    for year in [2023,2024]:
        print('Aggregate sharing',year,flush=True)
        ds.append(pd.read_parquet(CACHE/f'trips-{year}.parquet'))
    t=pd.concat(ds,ignore_index=True)
    c=pd.read_parquet(CACHE/'counters.parquet')
    sites=c.groupby('site')[['x','y']].mean()
    summary=[]
    for (year,kind,reg),g in t.groupby(['year','type','regime']):
        summary.append({'year':year,'type':kind,'regime':reg,'n':len(g),'distance_median':g.distance.median(),
                        'duration_median':g.duration.median(),'chord_median':g.chord.median(),
                        'short_chord_share':(g.chord<.1).mean(),'distance_below_chord_share':(g.distance+0.05<g.chord).mean()})
    pd.DataFrame(summary).to_csv(OUT/'trip-summary.csv',index=False)
    draw_trip_profiles(t)
    # OD maps are straight connectors, explicitly not estimated routes.
    od=t.groupby(['year','regime','ogx','ogy','dgx','dgy']).size().rename('n').reset_index()
    od.to_parquet(CACHE/'od-grid-counts.parquet',index=False)
    od.loc[od.n>=10].to_csv(OUT/'od-grid-counts.csv',index=False)
    # Distributional differences on a common grid. JS distance is bounded [0,1].
    probs={}
    for key,g in od.groupby(['year','regime']):
        probs[key]=g.set_index(['ogx','ogy','dgx','dgy']).n
    base=pd.concat(probs,axis=1).fillna(0);js=[]
    for a,b in [((2023,'day'),(2023,'night')),((2024,'day'),(2024,'night')),((2023,'day'),(2024,'day')),((2023,'night'),(2024,'night'))]:
        js.append({'a':str(a),'b':str(b),'js_distance':float(jensenshannon(base[a],base[b],base=2))})
    dump('od-distances.json',js)
    # Export only aggregated site/date/regime features, never endpoint records.
    keys=['date','regime']
    global_n=t.groupby(keys).size().rename('global_trips')
    grouped_type=t.groupby(keys+['type']).size().unstack(fill_value=0)
    global_features=pd.concat([global_n,grouped_type],axis=1)
    global_features['electric_share']=global_features.electric/global_features.global_trips
    coverage=pd.concat([pd.read_csv(OUT/f'sharing-period-coverage-{year}.csv') for year in [2023,2024]])
    # Pool the boundary service periods split across calendar-year input files.
    coverage=coverage.groupby(keys)[['endpoint_candidates','duration_eligible_endpoint']].sum()
    coverage['usable_for_comparison']=~((coverage.endpoint_candidates>=100)&
        (coverage.duration_eligible_endpoint<.01*coverage.endpoint_candidates))
    coverage.to_csv(OUT/'sharing-period-coverage.csv')
    draw_sharing_maps(t,sites,coverage.reset_index())
    global_features=global_features.join(coverage[['usable_for_comparison']])
    global_features=global_features.loc[global_features.usable_for_comparison].drop(columns='usable_for_comparison')
    features=[]
    for site,s in sites.iterrows():
        print('Feature proxy',site,flush=True)
        ox=t.ox.to_numpy()-s.x;oy=t.oy.to_numpy()-s.y
        dx=t.dx.to_numpy()-s.x;dy=t.dy.to_numpy()-s.y
        vx=dx-ox;vy=dy-oy
        u=-(ox*vx+oy*vy)/np.maximum(vx*vx+vy*vy,1e-10)
        perp=np.hypot(ox+np.clip(u,0,1)*vx,oy+np.clip(u,0,1)*vy)
        inward=(vx*(-s.x)+vy*(-s.y))>0
        f=t[keys].copy()
        f['starts_500m']=(ox*ox+oy*oy<.25).astype('int32')
        f['ends_500m']=(dx*dx+dy*dy<.25).astype('int32')
        for rad in [100,250]:
            cross=(perp<rad/1000)&(u>0)&(u<1)&(t.chord.to_numpy()>=.1)
            f[f'chord_{rad}_in']=(cross&inward).astype('int32')
            f[f'chord_{rad}_out']=(cross&~inward).astype('int32')
        g=f.groupby(keys).sum().join(global_features,how='inner').reset_index()
        g['site']=site;g['x']=s.x;g['y']=s.y;g['radius']=np.hypot(s.x,s.y);features.append(g)
    pd.concat(features).to_parquet(CACHE/'site-features.parquet',index=False)
    # Small-cell summaries of source/sink persistence avoid interpreting low counts.
    stats={'retained_total':len(t),'night_share':float((t.regime=='night').mean()),
           'electric_share':float((t.type=='electric').mean()),'median_duration':float(t.duration.median()),
           'median_distance':float(t.distance.median()),'recorded_below_chord':float((t.distance+.05<t.chord).mean()),
           'within_cell_share':float(((t.ogx==t.dgx)&(t.ogy==t.dgy)).mean())}
    dump('descriptive.json',stats)

if __name__=='__main__':main()
