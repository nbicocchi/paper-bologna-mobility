"""Fixed-configuration spatial + temporal stress test; no held-out-site calibration."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from common import ROOT,OUT,CACHE,TEXT,SEED,metrics,dump

class UniformScale:
    """One nonnegative factor, fitted only on calibration count exposure."""
    def fit(self,X,y,sample_weight):
        x=np.asarray(X)[:,0];w=np.asarray(sample_weight)
        self.factor=float(np.sum(np.asarray(y)*w)/max(np.sum(x*w),1e-12))
        return self
    def predict(self,X):return np.asarray(X)[:,0]*self.factor

class RegularizedPoisson:
    """Log-linear rate model; preprocessing is fitted on training locations only."""
    def fit(self,X,y,sample_weight):
        categories=['dow','night','direction']
        numeric=[c for c in X.columns if c not in categories]
        self.transform=ColumnTransformer([
            ('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categories),
            ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric)])
        features=self.transform.fit_transform(X)
        self.regressor=PoissonRegressor(alpha=1.,max_iter=1000,tol=1e-7)
        self.regressor.fit(features,y,sample_weight=sample_weight)
        return self
    def predict(self,X):return self.regressor.predict(self.transform.transform(X))

def estimator(model):
    if model=='uniform_chord':return UniformScale()
    if model.startswith('poisson_'):return RegularizedPoisson()
    return HistGradientBoostingRegressor(loss='poisson',max_iter=120,max_leaf_nodes=7,
                learning_rate=.06,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=SEED)

def main():
    a=pd.read_csv(OUT/'counter-regimes.csv');f=pd.read_parquet(CACHE/'site-features.parquet')
    d=a.merge(f,on=['site','date','regime'],validate='one_to_one')
    weather=pd.read_csv(ROOT/'dataset/meteo-bologna-ilmeteo-2019-2025.csv',sep=';')
    weather=weather[['date','temperature_mean_c','precipitation_mm']].drop_duplicates('date')
    # Service-day weather is a deliberately coarse exposure proxy for its overnight part.
    d=d.merge(weather,on='date',how='left',validate='many_to_one')
    dt=pd.to_datetime(d.date);d['year']=dt.dt.year;d['dow']=dt.dt.weekday
    d['sin_year']=np.sin(2*np.pi*dt.dt.dayofyear/365.25);d['cos_year']=np.cos(2*np.pi*dt.dt.dayofyear/365.25)
    d['night']=(d.regime=='night').astype(int)
    for col in ['global_trips','starts_500m','ends_500m','electric','mechanical']:
        d[col]=d[col]/d.expected
    blocks=[]
    for direction in ['inward','outward']:
        z=d.copy();z['direction']=(direction=='outward')*1;z['rate']=z[direction]/z.expected
        for rad in [100,250]:
            z[f'chord_{rad}']=z[f'chord_{rad}_'+('in' if direction=='inward' else 'out')]/z.expected
        blocks.append(z)
    d=pd.concat(blocks,ignore_index=True)
    # Eligibility fixed using training-year coverage only; no performance-based filtering.
    ntrain=d.loc[d.year==2023].groupby('site').date.nunique()
    sites=ntrain.loc[ntrain>=60].index.tolist()
    d=d.loc[d.site.isin(sites)].copy()
    d.to_parquet(CACHE/'prediction-input.parquet',index=False)
    base=['x','y','radius','dow','sin_year','cos_year','night','direction','temperature_mean_c','precipitation_mm']
    specs={'uniform_chord':['chord_250'],'calendar_weather':base,
           'global_sharing':base+['global_trips','electric_share'],
           'local_endpoints':base+['global_trips','electric_share','starts_500m','ends_500m'],
            'chord_proxy':base+['global_trips','electric_share','starts_500m','ends_500m','chord_100','chord_250']}
    poisson_features={'poisson_base':'calendar_weather','poisson_global':'global_sharing',
        'poisson_local':'local_endpoints','poisson_sharing':'chord_proxy'}
    specs.update({model:specs[features] for model,features in poisson_features.items()})
    rows=[]
    for site in sites:
        train=(d.year==2023)&(d.site!=site)&d.site.isin(sites)
        tests={'spatial_2023':(d.year==2023)&(d.site==site),
               'spatiotemporal_2024':(d.year==2024)&(d.site==site)}
        print('Hold out',site,'train',train.sum(),flush=True)
        for model,cols in specs.items():
            est=estimator(model)
            est.fit(d.loc[train,cols],d.loc[train,'rate'],sample_weight=d.loc[train,'expected'])
            for split,mask in tests.items():
                if not mask.any():continue
                z=d.loc[mask,['site','date','regime','direction','expected','rate']].copy()
                z['prediction']=est.predict(d.loc[mask,cols]);z['model']=model;z['split']=split;rows.append(z)
    pred=pd.concat(rows,ignore_index=True)
    pred['observed_count']=pred.rate*pred.expected;pred['predicted_count']=pred.prediction*pred.expected
    pred.to_csv(OUT/'predictions.csv',index=False)
    scores=[]
    for keys,g in pred.groupby(['split','model','site','regime']):
        scores.append(dict(zip(['split','model','site','regime'],keys))|metrics(g.observed_count,g.predicted_count))
    s=pd.DataFrame(scores);s.to_csv(OUT/'prediction-scores.csv',index=False)
    summary=[]
    for (split,model),g in s.groupby(['split','model']):
        # Equal site contribution after pooling day/night counts within each site.
        sub=pred.loc[(pred.split==split)&(pred.model==model)]
        site_metrics=[metrics(v.observed_count,v.predicted_count) for _,v in sub.groupby('site')]
        summary.append({'split':split,'model':model,'sites':len(site_metrics),
                        'macro_wape':np.mean([x['wape'] for x in site_metrics]),
                        'macro_mae':np.mean([x['mae'] for x in site_metrics]),
                        **{'pooled_'+k:v for k,v in metrics(sub.observed_count,sub.predicted_count).items()}})
    pd.DataFrame(summary).to_csv(OUT/'prediction-summary.csv',index=False)
    # Paired site bootstrap, not an hourly IID confidence interval.
    boot=[];rng=np.random.default_rng(SEED)
    for split in pred.split.unique():
        sub=pred.loc[pred.split==split]; vals={}
        for model,g in sub.groupby('model'):
            vals[model]=pd.Series({site:metrics(v.observed_count,v.predicted_count)['wape'] for site,v in g.groupby('site')})
        frame=pd.DataFrame(vals).dropna()
        for model in specs:
            if model in ['calendar_weather','poisson_base']:continue
            comparator='poisson_base' if model.startswith('poisson_') else 'calendar_weather'
            delta=(frame[model]-frame[comparator]).to_numpy()
            b=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
            boot.append({'split':split,'model':model,'comparator':comparator,'delta_macro_wape':float(delta.mean()),'lo':float(np.quantile(b,.025)),'hi':float(np.quantile(b,.975)),'sites':len(delta)})
    dump('paired-bootstrap.json',boot)
    fig,axs=plt.subplots(2,2,figsize=(8,5.7))
    families=[('Boosting',['calendar_weather','global_sharing','local_endpoints','chord_proxy']),
        ('Poisson',['poisson_base','poisson_global','poisson_local','poisson_sharing'])]
    styles=[('No sharing','#0072B2','o'),('Global sharing','#D55E00','s'),
        ('Local endpoints','#228833','^'),('OD geometry','#AA3377','x'),
        ('Uniform scaling','#777777','d')]
    splits=['spatial_2023','spatiotemporal_2024']
    titles=['Spatial 2023','Spatial + year 2024']
    for column,(split,title) in enumerate(zip(splits,titles)):
        z=pred.loc[pred.split==split]; names=sorted(z.site.unique())
        maximum=0
        for row,(family,models) in enumerate(families):
            ax=axs[row,column]
            for model,(label,color,marker) in zip(models+['uniform_chord'],styles):
                v=[metrics(z.loc[(z.site==site)&(z.model==model),'observed_count'],z.loc[(z.site==site)&(z.model==model),'predicted_count'])['wape']*100 for site in names]
                maximum=max(maximum,max(v))
                ax.plot(range(len(names)),v,marker=marker,color=color,lw=.8,
                    ls='--' if model=='uniform_chord' else '-',label=label,ms=3)
            ax.set_xticks(range(len(names)),names,rotation=65,fontsize=7)
            ax.set(title=f'{family} · {title}',ylabel='Site WAPE (%)')
            ax.title.set_fontsize(8);ax.yaxis.label.set_fontsize(8);ax.tick_params(axis='y',labelsize=7)
        for ax in axs[:,column]:ax.set_ylim(0,maximum*1.08)
    handles,names=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=5,fontsize=7,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(TEXT/'figures/prediction.pdf',bbox_inches='tight');plt.close(fig)
    dump('prediction-design.json',{'legacy_sites':sites,'splits':splits,'features':specs,'families':dict(families),'feature_pairing':poisson_features,'train_year':2023,'configuration':{'loss':'poisson','iterations':120,'leaves':7,'learning_rate':.06,'min_leaf':40,'l2':10},'poisson_configuration':{'alpha':1.,'max_iter':1000,'tol':1e-7,'categorical':['dow','night','direction'],'numeric_preprocessing':'Training-only median imputation and standardisation','features':list(poisson_features)},'weather_missing_rows':int(d[base].isna().any(axis=1).sum()),'rows':len(d)})
    from error_analysis import analyse_errors
    analyse_errors(pred,d)

if __name__=='__main__':main()
