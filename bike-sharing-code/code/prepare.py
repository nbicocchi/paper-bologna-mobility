"""Audit and de-identify supplied 2023--2024 data; never change the originals."""
import hashlib
import sys
import numpy as np
import pandas as pd
from common import ROOT, OUT, CACHE, EXCLUDED_SITES, MIN_RIDE_SECONDS, ride_duration_mask, xy, clock, dump

def sharing():
    audits=[]
    for year in [2023,2024]:
        dest=CACHE/f'trips-{year}.parquet'
        path=ROOT/'dataset'/f'bike-sharing-{year}.csv'
        if dest.exists() and '--force' not in sys.argv:
            import json
            old=OUT/f'audit-sharing-{year}.json'
            if old.exists():
                a=json.loads(old.read_text())
                current=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
                if a.get('cohort_version')=='endpoint-min30-v3' and a['sha256']==current:
                    print('Reuse verified',dest.name,flush=True);continue
        print('Read sharing',year,flush=True)
        d=pd.read_csv(path,sep=';',decimal=',',encoding='utf-8-sig')
        raw=len(d); duplicate=d.duplicated(); labels=d['Tipo bici'].value_counts(dropna=False).to_dict()
        d=d.loc[~duplicate].copy()
        start=pd.to_datetime(d['Inizio noleggio'],format='%d/%m/%Y %H:%M:%S',errors='coerce')
        end=pd.to_datetime(d['Fine noleggio'],format='%d/%m/%Y %H:%M:%S',errors='coerce')
        t=pd.DataFrame({'start':start,'end':end,'type':d['Tipo bici'].astype(str).str.lower().map(
            {'e-bike':'electric','ebike':'electric','bike':'mechanical','muscolare':'mechanical','muscolari':'mechanical','normal bike':'mechanical'})})
        for new,old in [('olat','Latitudine Start'),('olon','Longitudine Start'),('dlat','Latitudine Stop'),('dlon','Longitudine Stop'),('distance','DISTANCE (Km)')]:
            t[new]=pd.to_numeric(d[old],errors='coerce')
        t['recorded_duration_seconds']=(end-start).dt.total_seconds()
        t['duration']=t.recorded_duration_seconds/60
        t['ox'],t['oy']=xy(t.olat,t.olon);t['dx'],t['dy']=xy(t.dlat,t.dlon)
        t['chord']=np.hypot(t.dx-t.ox,t.dy-t.oy)
        flags={
            'invalid_time':start.isna()|(start.dt.year!=year),
            'duration_outside_0_5_180_min':~t.duration.between(.5,180),
            'duration_below_30_seconds':t.recorded_duration_seconds<MIN_RIDE_SECONDS,
            'missing_duration':t.recorded_duration_seconds.isna(),
            'outside_analysis_box':~(t.olat.between(44.40,44.60)&t.dlat.between(44.40,44.60)&t.olon.between(11.20,11.50)&t.dlon.between(11.20,11.50)),
            'distance_outside_0_50_km':~t.distance.between(0,50),
            'unknown_type':t.type.isna(),
            'chord_speed_over_45_kmh':t.chord/t.duration*60>45,
            'chord_under_100m':t.chord<.1}
        endpoint_good=~np.column_stack([flags[k] for k in ['invalid_time','outside_analysis_box','unknown_type','chord_under_100m']]).any(axis=1)
        eligible_duration=ride_duration_mask(t.recorded_duration_seconds)
        # User-requested cohort restriction, including zero/negative recorded durations.
        good=endpoint_good&eligible_duration
        duration_valid=t.duration.between(.5,180)&(t.chord/t.duration*60<=45)&end.notna()
        distance_valid=t.distance.between(0,50)
        audit={'year':year,'cohort_version':'endpoint-min30-v3','minimum_ride_seconds':MIN_RIDE_SECONDS,
               'raw':raw,'exact_duplicates':int(duplicate.sum()),'labels':labels,
               'flags_nonexclusive':{k:int(v.sum()) for k,v in flags.items()},'retained':int(good.sum()),
               'endpoint_valid_before_duration':int(endpoint_good.sum()),
               'removed_by_duration_from_endpoint_cohort':int((endpoint_good&~eligible_duration).sum()),
               'start_min':str(start.min()),'start_max':str(start.max()),
               'valid_duration_retained':int((good&duration_valid).sum()),
               'valid_distance_retained':int((good&distance_valid).sum()),
               'zero_duration_all_unique':int((t.duration==0).sum()),
               'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
        monthly=pd.DataFrame({'month':start.dt.month,'unique_records':1,'endpoint_valid':endpoint_good.astype(int),
            'retained':good.astype(int),'duration_below_30_seconds':flags['duration_below_30_seconds'].astype(int),
            'zero_duration':(t.duration==0).astype(int),'valid_duration':duration_valid.astype(int)})
        monthly.groupby('month').sum().to_csv(OUT/f'audit-monthly-{year}.csv')
        dates,regimes=clock(start)
        coverage=pd.DataFrame({'date':dates,'regime':regimes,'endpoint_candidates':endpoint_good.astype(int),
            'duration_eligible_endpoint':(endpoint_good&eligible_duration).astype(int)})
        coverage=coverage.groupby(['date','regime']).sum().reset_index()
        # Do not mistake a near-total recording defect for genuinely absent rentals.
        coverage['usable_for_comparison']=~((coverage.endpoint_candidates>=100)&
            (coverage.duration_eligible_endpoint<.01*coverage.endpoint_candidates))
        coverage.to_csv(OUT/f'sharing-period-coverage-{year}.csv',index=False)
        t['duration_valid']=duration_valid;t['distance_valid']=distance_valid
        t['duration']=t.duration.where(duration_valid);t['distance']=t.distance.where(distance_valid)
        t=t.loc[good].copy();t['date'],t['regime']=clock(t.start)
        t['year']=year;t['hour']=t.start.dt.hour;t['weekday']=t.start.dt.weekday
        t['weekend']=pd.to_datetime(t.date).dt.weekday>=5
        # Coarse fixed grid codes; fine gridding is descriptive, not released trajectories.
        for prefix in ['o','d']:
            t[prefix+'gx']=np.floor(t[prefix+'x']).astype('int16')
            t[prefix+'gy']=np.floor(t[prefix+'y']).astype('int16')
        t.to_parquet(dest,index=False)
        dump(f'audit-sharing-{year}.json',audit)
        print(audit,flush=True)

def counters():
    d=pd.read_csv(ROOT/'dataset/colonnine-conta-bici.csv',sep=';',encoding='utf-8-sig')
    d['time']=pd.to_datetime(d['Data'],utc=True)
    d=d.loc[d.time.dt.tz_convert('Europe/Rome').dt.year.isin([2023,2024])].copy()
    input_rows=len(d)
    d['site']=d['Dispositivo conta-bici'].str.replace(r'_(I|II)$','',regex=True)
    excluded=d.site.isin(EXCLUDED_SITES)
    excluded_rows=int(excluded.sum())
    dump('counter-selection.json',{'excluded_sites':sorted(EXCLUDED_SITES),
        'input_rows_2023_2024':input_rows,'excluded_rows':excluded_rows,
        'excluded_site_rows':d.loc[excluded].groupby('site').size().to_dict(),
        'selection_basis':'User-requested exclusion, not automatic fault or performance classification.'})
    d=d.loc[~excluded].copy()
    duplicates=int(d.duplicated(['time','Dispositivo conta-bici']).sum());conflicting_rows=0
    if duplicates:
        # Only identical duplicates may be removed without an adjudication rule.
        d=d.drop_duplicates()
        conflict=d.duplicated(['time','Dispositivo conta-bici'],keep=False)
        conflicting_rows=int(conflict.sum())
        d.loc[conflict].to_csv(OUT/'counter-conflicting-duplicates.csv',index=False)
        # Without provenance, neither average nor arbitrarily choose competing records.
        d=d.loc[~conflict].copy()
    d=d.rename(columns={'Dispositivo conta-bici':'device','Direzione centro':'inward','Direzione periferia':'outward','Totale':'total'})
    d['site']=d.device.str.replace(r'_(I|II)$','',regex=True)
    loc=d['Geo Point'].str.split(',',expand=True).astype(float)
    d['x'],d['y']=xy(loc[0],loc[1])
    d=d.sort_values(['device','time'])
    # Conservative failure screen: contiguous all-zero runs >=48 real hours.
    zero=(d.total==0)&(d.inward==0)&(d.outward==0)
    breaks=(d.device!=d.device.shift())|(~zero)|(~zero.shift(fill_value=False))|(d.time.diff()!=pd.Timedelta(hours=1))
    run=breaks.cumsum();length=zero.groupby(run).transform('sum')
    d['suspect_zero']=zero&(length>=48)
    d['valid']=(~d.suspect_zero)&d[['inward','outward','total']].notna().all(axis=1)&(d[['inward','outward','total']]>=0).all(axis=1)&(d.total==d.inward+d.outward)
    d['date'],d['regime']=clock(d.time.dt.tz_convert('Europe/Rome'))
    audit=d.groupby('device').agg(site=('site','first'),n=('time','size'),valid=('valid','sum'),suspect_zero=('suspect_zero','sum'),zero=('total',lambda s:(s==0).mean()),mean=('total','mean'),inward=('inward','sum'),outward=('outward','sum'),x=('x','first'),y=('y','first'))
    audit.to_csv(OUT/'audit-counters.csv')
    dump('audit-counters.json',{'input_rows':input_rows,'excluded_rows':excluded_rows,'rows':len(d),'duplicate_keys':duplicates,'conflicting_rows_excluded':conflicting_rows,'sites':d.site.nunique(),'devices':d.device.nunique(),'total_mismatches':int((d.total!=d.inward+d.outward).sum()),'suspect_zero_rows':int(d.suspect_zero.sum())})
    d.to_parquet(CACHE/'counters.parquet',index=False)
    # Every constituent must be available and valid at the same timestamp.
    expected_devices=d.groupby('site').device.nunique()
    h=d.groupby(['site','time']).agg(n=('device','nunique'),valid=('valid','all'),inward=('inward','sum'),outward=('outward','sum'))
    h['valid']=h.valid&(h.n==h.index.get_level_values('site').map(expected_devices))
    h=h.reset_index();h['date'],h['regime']=clock(h.time.dt.tz_convert('Europe/Rome'))
    grid=pd.DataFrame({'time':pd.date_range('2023-01-01 06:00','2025-01-01 06:00',freq='h',tz='Europe/Rome',inclusive='left')})
    grid['date'],grid['regime']=clock(grid.time)
    expected=grid.groupby(['date','regime']).size().rename('expected')
    a=h.loc[h.valid].groupby(['site','date','regime']).agg(hours=('time','size'),inward=('inward','sum'),outward=('outward','sum')).join(expected)
    a=a.loc[a.hours==a.expected].reset_index()
    # Last service day requires 2025 observations not contained in the audit slice.
    a=a.loc[(a.date>='2023-01-01')&(a.date<'2024-12-31')]
    a.to_csv(OUT/'counter-regimes.csv',index=False)
    print('Counters',len(d),'rows;',len(a),'complete site/service-day/regimes',flush=True)

if __name__=='__main__':
    sharing();counters()
