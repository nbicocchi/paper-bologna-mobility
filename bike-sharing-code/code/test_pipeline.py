"""Small deterministic tests, plus checks on generated artifacts when available."""
import unittest
import numpy as np
import pandas as pd
from common import OUT,CACHE,EXCLUDED_SITES,ride_duration_mask,clock,metrics
from synthetic import network,aggregate,fit
from citymap import six_hour_blocks,six_hour_balances,leading_od_links,map_periods,day_night_grid,map_windows,window_map_periods,window_map_grid

class PipelineTests(unittest.TestCase):
    def test_service_day(self):
        d=pd.Series(pd.to_datetime(['2024-01-06 02:00','2024-01-06 06:00','2024-01-06 23:00']))
        date,reg=clock(d)
        self.assertEqual(list(date),['2024-01-05','2024-01-06','2024-01-06'])
        self.assertEqual(list(reg),['night','day','night'])
    def test_dst(self):
        for start,end,expected in [('2024-03-30 23:00','2024-03-31 06:00',6),('2024-10-26 23:00','2024-10-27 06:00',8)]:
            self.assertEqual(len(pd.date_range(start,end,freq='h',tz='Europe/Rome',inclusive='left')),expected)
    def test_metrics_zero(self):
        self.assertEqual(metrics([0,0],[0,0])['wape'],0)
    def test_direction_aggregation(self):
        *_,mats=network();A=mats[0]
        self.assertTrue(np.array_equal(aggregate(A,[1,3],False),aggregate(A,[1,3],True).reshape(2,2,-1).sum(axis=1)))
    def test_inverse_positive(self):
        A=np.array([[1.,0.,1.],[0.,1.,1.]])
        p=np.ones(3)*3;y=np.array([20.,10.])
        q,ok,_=fit(p,A,y)
        self.assertTrue(ok);self.assertTrue((q>0).all())
        self.assertLess(np.sum(abs(A@q-y)),np.sum(abs(A@p-y)))
    def test_no_identity_in_cache(self):
        p=CACHE/'trips-2023.parquet'
        if not p.exists():self.skipTest('No real-data cache')
        import pyarrow.parquet as pq
        self.assertFalse({'Codice utente','BikeID'}&set(pq.read_schema(p).names))
    def test_no_target_feature(self):
        import json
        p=OUT/'prediction-design.json'
        if not p.exists():self.skipTest('Prediction not run')
        for cols in json.loads(p.read_text())['features'].values():
            self.assertFalse({'rate','inward','outward','observed_count','total'}&set(cols))
        d=pd.read_parquet(CACHE/'prediction-input.parquet')
        # y must remain a geographic coordinate, never the outcome.
        self.assertLess(d.y.abs().max(),15)
    def test_artifact_keys(self):
        if not (OUT/'predictions.csv').exists():self.skipTest('Prediction not run')
        d=pd.read_csv(OUT/'predictions.csv')
        self.assertFalse(d.duplicated(['split','model','site','date','regime','direction']).any())
        self.assertTrue(np.isfinite(d.predicted_count).all());self.assertTrue((d.predicted_count>=0).all())
    def test_duration_threshold(self):
        seconds=[-1,0,29,29.999,30,30.001,60,np.nan,np.inf]
        self.assertEqual(ride_duration_mask(seconds).tolist(),[False,False,False,False,True,True,True,False,False])
    def test_retained_durations(self):
        for year in [2023,2024]:
            p=CACHE/f'trips-{year}.parquet'
            if not p.exists():self.skipTest('Trip preparation not run')
            seconds=pd.read_parquet(p,columns=['recorded_duration_seconds']).recorded_duration_seconds
            self.assertTrue(ride_duration_mask(seconds).all())
    def test_october_filter_is_audited(self):
        p=OUT/'audit-monthly-2023.csv'
        if not p.exists():self.skipTest('Audit not run')
        r=pd.read_csv(p).set_index('month').loc[10]
        self.assertGreater(r.zero_duration/r.unique_records,.99)
        self.assertGreater(r.endpoint_valid,200000)
        self.assertLessEqual(r.retained,1)
    def test_count_cohort_integrity(self):
        import json
        p=OUT/'descriptive.json'
        if not p.exists():self.skipTest('Description not run')
        n=sum(json.loads((OUT/f'audit-sharing-{y}.json').read_text())['retained'] for y in [2023,2024])
        self.assertEqual(n,json.loads(p.read_text())['retained_total'])
    def test_excluded_counter_sites(self):
        p=OUT/'counter-regimes.csv'
        if not p.exists():self.skipTest('Counter preparation not run')
        for path in [p,OUT/'site-coverage.csv',OUT/'predictions.csv',OUT/'audit-counters.csv']:
            if path.exists():
                self.assertFalse(set(pd.read_csv(path).site)&EXCLUDED_SITES,str(path))
        for path in [CACHE/'counters.parquet',CACHE/'site-features.parquet',CACHE/'prediction-input.parquet']:
            if path.exists():self.assertFalse(set(pd.read_parquet(path,columns=['site']).site)&EXCLUDED_SITES,str(path))
    def test_six_hour_boundaries(self):
        t=pd.Series(pd.to_datetime(['2024-01-01 00:00','2024-01-01 05:59',
            '2024-01-01 06:00','2024-01-01 11:59','2024-01-01 12:00',
            '2024-01-01 17:59','2024-01-01 18:00','2024-01-01 23:59']))
        self.assertEqual(six_hour_blocks(t).tolist(),[0,0,1,1,2,2,3,3])
    def test_arrivals_use_return_time(self):
        t=pd.DataFrame({'start':pd.to_datetime(['2024-01-01 05:59','2024-01-01 23:59','2023-12-31 23:59']),
            'end':pd.to_datetime(['2024-01-01 06:01','2024-01-02 00:01','2024-01-01 00:01']),
            'ox':[0.,0.,0.],'oy':[0.,0.,0.],'dx':[.5,.5,.5],'dy':[0.,0.,0.]})
        b=six_hour_balances(t)
        self.assertEqual(int(b.departures.sum()),2)
        self.assertEqual(int(b.arrivals.sum()),3)
        self.assertEqual(int(b.loc[b.block==1,'arrivals'].sum()),1)
        self.assertEqual(int(b.loc[b.block==0,'arrivals'].sum()),2)
        self.assertEqual(int(b.loc[b.block==3,'departures'].sum()),1)
    def test_city_balance_map(self):
        import json
        p=OUT/'city-balance-map.json'
        if not p.exists():self.skipTest('City map not generated')
        info=json.loads(p.read_text())
        self.assertEqual(info['cell_km'],.5)
        self.assertFalse(set(info['counter_sites'])&EXCLUDED_SITES)
        self.assertEqual(len(info['blocks']),4)
        self.assertEqual([b['block'] for b in info['blocks']],[0,1,2,3])
        grid=pd.read_csv(OUT/'grid-balances.csv')
        self.assertTrue(grid.activity.eq(grid.arrivals+grid.departures).all())
        np.testing.assert_allclose(grid.ratio,(grid.arrivals-grid.departures)/grid.activity)
        self.assertTrue(grid.ratio.between(-1,1).all())
        trips=pd.concat([pd.read_parquet(CACHE/f'trips-{y}.parquet',columns=['start','end']) for y in [2023,2024]])
        self.assertEqual(int(grid.departures.sum()),int((trips.start.dt.year==2024).sum()))
        self.assertEqual(int(grid.arrivals.sum()),int((trips.end.dt.year==2024).sum()))
    def test_od_map_link_selection(self):
        od=pd.DataFrame({'year':[2023,2023,2023,2023,2024],
            'regime':['day','day','day','night','day'],
            'ogx':[0,0,0,0,0],'ogy':[0,0,0,0,0],
            'dgx':[0,1,2,1,1],'dgy':[0,0,0,0,0],
            'n':[100,20,5,30,40]})
        links,total=leading_od_links(od,2023,'day')
        self.assertEqual(total,125)
        self.assertEqual(links.n.tolist(),[20])
        self.assertAlmostEqual(links.share.iloc[0],20/125)
        self.assertTrue(leading_od_links(od,2024,'night')[0].empty)
    def test_city_od_map(self):
        import json
        p=OUT/'city-od-map.json'
        if not p.exists():self.skipTest('OD map not generated')
        info=json.loads(p.read_text())
        self.assertEqual(info['cell_km'],1)
        self.assertEqual([(p['year'],p['regime']) for p in info['panels']],
            [(2023,'day'),(2023,'night'),(2024,'day'),(2024,'night')])
        self.assertTrue(all(p['displayed_links']==40 for p in info['panels']))
        self.assertFalse(set(info['counter_sites'])&EXCLUDED_SITES)
        od=pd.read_parquet(CACHE/'od-grid-counts.parquet')
        for panel in info['panels']:
            links,total=leading_od_links(od,panel['year'],panel['regime'])
            self.assertEqual(panel['retained_trips'],total)
            self.assertEqual(panel['displayed_trips'],int(links.n.sum()))
    def test_compromised_periods_not_predictors(self):
        p=OUT/'sharing-period-coverage.csv'
        if not p.exists():self.skipTest('Sharing coverage not generated')
        bad=pd.read_csv(p).query('not usable_for_comparison')[['date','regime']]
        features=pd.read_parquet(CACHE/'site-features.parquet',columns=['date','regime'])
        self.assertEqual(len(features.merge(bad,on=['date','regime'])),0)
    def test_map_periods_dst_and_filtering(self):
        coverage=pd.DataFrame({'date':['2024-03-30','2024-10-26','2023-10-02',
            '2024-12-31','2024-12-31'],
            'regime':['night','night','day','day','night'],
            'usable_for_comparison':[True,True,False,True,True]})
        periods=map_periods(coverage).set_index(['date','regime'])
        self.assertEqual(periods.loc[('2024-03-30','night'),'hours'],6)
        self.assertEqual(periods.loc[('2024-10-26','night'),'hours'],8)
        self.assertEqual(periods.loc[('2024-12-31','day'),'hours'],17)
        self.assertNotIn(('2024-12-31','night'),periods.index)
        self.assertNotIn(('2023-10-02','day'),periods.index)
    def test_day_night_events_use_their_own_times(self):
        t=pd.DataFrame({'start':pd.to_datetime(['2024-01-01 22:59','2024-01-02 05:59','2024-01-03 10:00']),
            'end':pd.to_datetime(['2024-01-01 23:01','2024-01-02 06:01','2024-01-03 10:10']),
            'ox':[0.,0.,0.],'oy':[0.,0.,0.],'dx':[0.,0.,0.],'dy':[0.,0.,0.]})
        periods=pd.DataFrame({'date':['2024-01-01','2024-01-01'],
            'regime':['day','night'],'hours':[17,7]})
        grid=day_night_grid(t,periods).set_index('regime')
        self.assertEqual(int(grid.departures.sum()),2)
        self.assertEqual(int(grid.arrivals.sum()),1)
        self.assertAlmostEqual(grid.loc['day','events_per_hour'],1/17)
        self.assertAlmostEqual(grid.loc['night','events_per_hour'],2/7)
        self.assertEqual(grid.loc['day','ratio'],-1)
        self.assertEqual(grid.loc['night','ratio'],0)
    def test_matched_activity_balance_maps(self):
        import json
        p=OUT/'sharing-map-design.json'
        if not p.exists():self.skipTest('Paired sharing maps not generated')
        info=json.loads(p.read_text())
        self.assertEqual(info['years'],[2023,2024])
        self.assertEqual(info['cell_km'],.5)
        self.assertEqual([p['block'] for p in info['panels']],[0,1,2,3])
        self.assertFalse(set(info['counter_sites'])&EXCLUDED_SITES)
        grid=pd.read_csv(OUT/'sharing-map-grid.csv')
        periods=pd.read_csv(OUT/'sharing-map-periods.csv')
        self.assertTrue(grid.activity.eq(grid.arrivals+grid.departures).all())
        np.testing.assert_allclose(grid.ratio,(grid.arrivals-grid.departures)/grid.activity)
        np.testing.assert_allclose(grid.events_per_hour,grid.activity/grid.hours)
        expected=window_map_periods(pd.read_csv(OUT/'sharing-period-coverage.csv'))
        pd.testing.assert_frame_equal(periods.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False)
        for panel in info['panels']:
            data=grid.loc[grid.block==panel['block']]
            self.assertEqual(panel['hours'],int(periods.loc[periods.block==panel['block'],'hours'].sum()))
            self.assertEqual(panel['arrivals'],int(data.arrivals.sum()))
            self.assertEqual(panel['departures'],int(data.departures.sum()))
    def test_window_map_coverage(self):
        coverage=pd.DataFrame({'date':['2024-03-30','2024-03-30','2024-03-31','2024-03-31',
            '2024-10-26','2024-10-26','2024-12-31','2024-12-31'],
            'regime':['day','night','day','night','day','night','day','night'],
            'usable_for_comparison':[True,True,True,False,True,True,True,True]})
        periods=window_map_periods(coverage).set_index(['date','block'])
        self.assertEqual(periods.loc[('2024-03-30',3),'hours'],9)
        self.assertEqual(periods.loc[('2024-10-26',3),'hours'],11)
        self.assertEqual(periods.loc[('2024-03-31',0),'hours'],4)
        self.assertEqual(periods.loc[('2024-03-31',1),'hours'],5)
        self.assertEqual(periods.loc[('2024-03-31',2),'hours'],5)
        self.assertNotIn(('2024-03-31',3),periods.index)
        self.assertNotIn(('2024-12-31',3),periods.index)
    def test_window_map_boundaries(self):
        times=pd.Series(pd.to_datetime(['2024-01-01 '+hour for hour in
            ['05:59','06:00','09:59','10:00','14:59','15:00','19:59','20:00']]))
        self.assertEqual(map_windows(times).tolist(),[3,0,0,1,1,2,2,3])
    def test_window_map_event_times(self):
        t=pd.DataFrame({'start':pd.to_datetime(['2024-01-01 09:59','2024-01-01 19:59','2024-01-02 05:59']),
            'end':pd.to_datetime(['2024-01-01 10:01','2024-01-01 20:01','2024-01-02 06:01']),
            'ox':[0.,0.,0.],'oy':[0.,0.,0.],'dx':[0.,0.,0.],'dy':[0.,0.,0.]})
        periods=pd.DataFrame({'date':['2024-01-01']*4,'block':[0,1,2,3],'hours':[4,5,5,10]})
        grid=window_map_grid(t,periods).set_index('block')
        self.assertEqual(grid.departures.tolist(),[1,0,1,1])
        self.assertEqual(grid.arrivals.tolist(),[0,1,0,1])
        np.testing.assert_allclose(grid.events_per_hour,[1/4,1/5,1/5,2/10])
        self.assertEqual(grid.ratio.tolist(),[-1,1,-1,0])
    def test_poisson_preprocessing_uses_training_only(self):
        from predict import RegularizedPoisson
        x=pd.DataFrame({'dow':[0,1,0,1,0,1,0,1],
            'night':[0,0,1,1,0,0,1,1],'direction':[0,1,0,1,0,1,0,1],
            'x':np.arange(8,dtype=float)})
        model=RegularizedPoisson().fit(x,np.arange(1,9),np.ones(8))
        scaler=model.transform.named_transformers_['numeric'].named_steps['standardscaler']
        mean=scaler.mean_.copy()
        test=x.copy();test['x']=100.;test['dow']=6
        prediction=model.predict(test)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction>=0).all())
        np.testing.assert_allclose(scaler.mean_,mean)
        np.testing.assert_allclose(mean,[3.5])
    def test_poisson_prediction_keys_match(self):
        if not (OUT/'predictions.csv').exists():self.skipTest('Prediction not run')
        p=pd.read_csv(OUT/'predictions.csv')
        keys=['split','site','date','regime','direction']
        base=p.loc[p.model=='calendar_weather',keys].sort_values(keys).reset_index(drop=True)
        for model in ['poisson_base','poisson_global','poisson_local','poisson_sharing']:
            other=p.loc[p.model==model,keys].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(base,other)
    def test_symmetric_model_feature_design(self):
        import json
        path=OUT/'prediction-design.json'
        if not path.exists():self.skipTest('Prediction not run')
        design=json.loads(path.read_text())
        self.assertEqual(len(design['features']),9)
        self.assertEqual(len(design['families']['Boosting']),4)
        self.assertEqual(len(design['families']['Poisson']),4)
        self.assertEqual(design['splits'],['spatial_2023','spatiotemporal_2024'])
        for poisson,boosting in design['feature_pairing'].items():
            self.assertEqual(design['features'][poisson],design['features'][boosting])
        boot=json.loads((OUT/'paired-bootstrap.json').read_text())
        for comparison in boot:
            expected='poisson_base' if comparison['model'].startswith('poisson_') else 'calendar_weather'
            self.assertEqual(comparison['comparator'],expected)
    def test_site_error_diagnostics(self):
        path=OUT/'prediction-site-diagnostics.csv'
        if not path.exists():self.skipTest('Site diagnostics not generated')
        d=pd.read_csv(path)
        self.assertFalse(d.duplicated(['split','site']).any())
        self.assertEqual(len(d),8)
        self.assertEqual(set(d.split),{'spatial_2023','spatiotemporal_2024'})
        np.testing.assert_allclose(d.chord_gain_pp,
            100*(d.calendar_weather_wape-d.chord_proxy_wape))
        self.assertTrue((d.observed_rate>0).all())
        weekly=pd.read_csv(OUT/'prediction-weekly-2024.csv')
        self.assertEqual(weekly.site.nunique(),4)
        self.assertTrue((weekly.hours>0).all())
        np.testing.assert_allclose(weekly.observed_rate,weekly.observed/weekly.hours)
        np.testing.assert_allclose(weekly.predicted_rate,weekly.predicted/weekly.hours)

if __name__=='__main__':unittest.main()
