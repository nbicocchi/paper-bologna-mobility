"""Location-level error diagnostics and out-of-location 2024 time profiles."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import OUT,TEXT,metrics

def analyse_errors(pred,inputs):
    rows=[]
    for (split,site),g in pred.groupby(['split','site']):
        base=g.loc[g.model=='calendar_weather']
        year=2023 if split=='spatial_2023' else 2024
        context=inputs.loc[(inputs.site==site)&(inputs.year==year)]
        observed=base.observed_count.sum()
        row={'split':split,'site':site,'directional_periods':len(base),
            'observed_count':observed,'observed_rate':observed/base.expected.sum(),
            'endpoint_events_per_hour':np.average(context.starts_500m+context.ends_500m,
                weights=context.expected)}
        for model,values in g.groupby('model'):
            result=metrics(values.observed_count,values.predicted_count)
            row[model+'_wape']=result['wape'];row[model+'_bias']=result['bias']
        row['chord_gain_pp']=100*(row['calendar_weather_wape']-row['chord_proxy_wape'])
        row['local_gain_pp']=100*(row['calendar_weather_wape']-row['local_endpoints_wape'])
        row['poisson_gain_pp']=100*(row['poisson_base_wape']-row['poisson_sharing_wape'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'prediction-site-diagnostics.csv',index=False)

    # Sum directions at each complete period before pooling into weekly hourly rates.
    selected=pred.loc[pred.split=='spatiotemporal_2024']
    periods=selected.groupby(['model','site','date','regime'],as_index=False).agg(
        observed=('observed_count','sum'),predicted=('predicted_count','sum'),
        hours=('expected','first'))
    periods['week']=pd.to_datetime(periods.date).dt.to_period('W-SUN').dt.start_time
    weekly=periods.groupby(['model','site','week'],as_index=False)[['observed','predicted','hours']].sum()
    weekly['observed_rate']=weekly.observed/weekly.hours
    weekly['predicted_rate']=weekly.predicted/weekly.hours
    weekly.to_csv(OUT/'prediction-weekly-2024.csv',index=False)
    styles={'calendar_weather':('No sharing','#777777','--'),
        'local_endpoints':('Local endpoints','#D55E00',':'),
        'chord_proxy':('OD geometry','#0072B2','-'),
        'poisson_base':('Poisson, no sharing','#228833','--'),
        'poisson_sharing':('Poisson + sharing','#8c4b99','-.')}
    sites=sorted(weekly.site.unique())
    fig,axs=plt.subplots(2,2,figsize=(9,5.5))
    for ax,site in zip(axs.flat,sites):
        data=weekly.loc[weekly.site==site]
        actual=data.loc[data.model=='calendar_weather'].set_index('week').sort_index()
        dates=pd.date_range(actual.index.min(),actual.index.max(),freq='W-MON')
        actual=actual.reindex(dates)
        ax.plot(actual.index,actual.observed_rate,color='black',lw=1.1,label='Observed')
        for model,(label,color,ls) in styles.items():
            values=data.loc[data.model==model].set_index('week').reindex(dates)
            ax.plot(values.index,values.predicted_rate,label=label,color=color,ls=ls,lw=.85)
        ax.set(title=site,ylabel='Bicycles / hour',xlim=(pd.Timestamp('2024-01-01'),pd.Timestamp('2024-12-31')),ylim=(0,None))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(labelsize=7);ax.title.set_fontsize(9);ax.yaxis.label.set_fontsize(8)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7,frameon=False)
    fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(TEXT/'figures/observed-predicted.pdf',bbox_inches='tight')
    fig.savefig(TEXT/'figures/observed-predicted.png',bbox_inches='tight',dpi=180)
    plt.close(fig)
