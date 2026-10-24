"""Sharing activity and balances over a cached OpenStreetMap background.

The background is present-day context, never the historical routing network.
Only tiles needed by this single map viewport are requested, sequentially/cached.
No rental coordinates or identifiers are sent to the tile service.
"""
import datetime
import hashlib
import io
import json
import math
import numpy as np
import pandas as pd
import requests
from PIL import Image
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.lines import Line2D
from common import CACHE, OUT, TEXT, LAT0, LON0, clock, dump

ATTRIBUTION = 'Map: © OpenStreetMap contributors · openstreetmap.org/copyright'
TO_MERCATOR = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

def project_local(x, y):
    lon=LON0+np.asarray(x)/(111.195*np.cos(np.deg2rad(LAT0)))
    lat=LAT0+np.asarray(y)/111.195
    return TO_MERCATOR.transform(lon,lat)

def tile_index(lon,lat,z):
    n=2**z
    return int((lon+180)/360*n),int((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)

def basemap(bounds,zoom=12):
    """Return RGB mosaic and exact Web-Mercator extent, failing if no real map."""
    west,south,east,north=bounds
    x0,y1=tile_index(west,south,zoom);x1,y0=tile_index(east,north,zoom)
    folder=CACHE/'basemap';folder.mkdir(exist_ok=True)
    mosaic=Image.new('RGB',((x1-x0+1)*256,(y1-y0+1)*256))
    tiles=[]
    session=requests.Session()
    session.headers['User-Agent']='BolognaCyclingResearch/0.1 (local scientific manuscript; cached map viewport)'
    for x in range(x0,x1+1):
        for y in range(y0,y1+1):
            name=f'osm-{zoom}-{x}-{y}.png';path=folder/name
            url=f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png'
            if not path.exists():
                response=session.get(url,timeout=45);response.raise_for_status()
                image=Image.open(io.BytesIO(response.content));image.load()
                if image.size!=(256,256):raise ValueError('Unexpected basemap tile geometry')
                path.write_bytes(response.content)
            image=Image.open(path).convert('RGB')
            mosaic.paste(image,((x-x0)*256,(y-y0)*256))
            tiles.append({'file':name,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    half=math.pi*6378137.;size=2*half/2**zoom
    extent=(x0*size-half,(x1+1)*size-half,half-(y1+1)*size,half-y0*size)
    dump('city-basemap.json',{'provider':'OpenStreetMap standard raster tiles','zoom':zoom,
        'attribution':ATTRIBUTION,'license_url':'https://www.openstreetmap.org/copyright',
        'recorded_utc':datetime.datetime.now(datetime.UTC).isoformat(),'tiles':tiles,
        'purpose':'Cartographic context only; not a 2023–2024 routing graph.'})
    # Lighten the context so shared count colours are legible without hiding streets.
    rgb=np.asarray(mosaic)/255.
    gray=np.sum(rgb*np.array([.299,.587,.114]),axis=2)
    context=np.repeat((.6*gray+.4)[...,None],3,axis=2)
    return context,extent

def six_hour_blocks(timestamps):
    """Local half-open bins [00,06), [06,12), [12,18), [18,24)."""
    return timestamps.dt.hour//6

BLOCK_LABELS=['00:00–06:00','06:00–12:00','12:00–18:00','18:00–24:00']

def map_periods(coverage):
    """Complete service periods in 2023–2024 with usable sharing coverage."""
    hours=pd.Series(pd.date_range('2023-01-01 06:00','2024-12-31 23:00',
        freq='h',tz='Europe/Rome',inclusive='left'))
    # Service dates follow the local wall clock; exposure retains actual DST hours.
    dates,regimes=clock(hours.dt.tz_localize(None))
    periods=pd.DataFrame({'date':dates,'regime':regimes}).groupby(
        ['date','regime']).size().rename('hours').reset_index()
    usable=coverage.loc[coverage.usable_for_comparison,['date','regime']]
    return periods.merge(usable,on=['date','regime'],validate='one_to_one')

def day_night_grid(t,periods,cell=.5):
    """Pool endpoint events in the same valid periods for both map metrics."""
    counts=[]
    for prefix,time,kind in [('o','start','departures'),('d','end','arrivals')]:
        date,regime=clock(t[time])
        events=pd.DataFrame({'date':date,'regime':regime,
            'gx':np.floor(t[prefix+'x']/cell).astype('int16'),
            'gy':np.floor(t[prefix+'y']/cell).astype('int16')})
        events=events.merge(periods[['date','regime']],on=['date','regime'],
            validate='many_to_one')
        counts.append(events.groupby(['regime','gx','gy']).size().rename(kind))
    grid=pd.concat(counts,axis=1).fillna(0).astype('int64').reset_index()
    grid['activity']=grid.arrivals+grid.departures
    grid['balance']=grid.arrivals-grid.departures
    grid['ratio']=grid.balance/grid.activity
    exposure=periods.groupby('regime').hours.sum()
    grid['hours']=grid.regime.map(exposure)
    grid['events_per_hour']=grid.activity/grid.hours
    grid['cell_km']=cell
    return grid

MAP_LABELS=['06:00–10:00','10:00–15:00','15:00–20:00','Night (20:00–06:00)']

def map_windows(timestamps):
    """Half-open 06–10, 10–15, 15–20, and overnight 20–06 windows."""
    hour=timestamps.dt.hour
    return np.select([(hour>=6)&(hour<10),(hour>=10)&(hour<15),
        (hour>=15)&(hour<20)],[0,1,2],default=3)

def window_map_periods(coverage):
    """Keep complete map windows only when every hour has usable coverage."""
    hours=pd.Series(pd.date_range('2023-01-01 06:00','2025-01-01 06:00',freq='h',
        tz='Europe/Rome',inclusive='left')).dt.tz_localize(None)
    service_date,regime=clock(hours)
    calendar=pd.DataFrame({'date':service_date,
        'block':map_windows(hours),'service_date':service_date,'regime':regime,
        'in_study':hours.dt.year.isin([2023,2024])})
    usable=coverage[['date','regime','usable_for_comparison']].rename(
        columns={'date':'service_date'})
    calendar=calendar.merge(usable,on=['service_date','regime'],how='left',validate='many_to_one')
    calendar['usable_for_comparison']=calendar.usable_for_comparison.fillna(False).astype(bool)&calendar.in_study
    periods=calendar.groupby(['date','block']).agg(hours=('block','size'),
        usable=('usable_for_comparison','all')).reset_index()
    return periods.loc[periods.usable,['date','block','hours']]

def window_map_grid(t,periods,cell=.5):
    """Assign endpoints by their own event time; after midnight stays with evening."""
    counts=[]
    for prefix,time,kind in [('o','start','departures'),('d','end','arrivals')]:
        date,_=clock(t[time])
        events=pd.DataFrame({'date':date,'block':map_windows(t[time]),
            'gx':np.floor(t[prefix+'x']/cell).astype('int16'),
            'gy':np.floor(t[prefix+'y']/cell).astype('int16')})
        events=events.merge(periods[['date','block']],on=['date','block'],validate='many_to_one')
        counts.append(events.groupby(['block','gx','gy']).size().rename(kind))
    grid=pd.concat(counts,axis=1).fillna(0).astype('int64').reset_index()
    grid=grid.sort_values(['block','gx','gy']).reset_index(drop=True)
    grid['activity']=grid.arrivals+grid.departures
    grid['balance']=grid.arrivals-grid.departures
    grid['ratio']=grid.balance/grid.activity
    grid['hours']=grid.block.map(periods.groupby('block').hours.sum())
    grid['events_per_hour']=grid.activity/grid.hours
    grid['cell_km']=cell
    grid['block_label']=grid.block.map(dict(enumerate(MAP_LABELS)))
    return grid

def draw_sharing_maps(t,sites,coverage):
    """Matched four-block activity and balance maps, pooled over 2023–2024."""
    cell=.5;minimum=200
    periods=window_map_periods(coverage)
    periods.to_csv(OUT/'sharing-map-periods.csv',index=False)
    grid=window_map_grid(t,periods,cell)
    grid.to_csv(OUT/'sharing-map-grid.csv',index=False)
    xedges=np.arange(-7,8+cell,cell);yedges=np.arange(-5,6+cell,cell)
    west=LON0+xedges[0]/(111.195*np.cos(np.deg2rad(LAT0)))
    east=LON0+xedges[-1]/(111.195*np.cos(np.deg2rad(LAT0)))
    south=LAT0+yedges[0]/111.195;north=LAT0+yedges[-1]/111.195
    context,extent=basemap((west,south,east,north))
    X,Y=np.meshgrid(xedges,yedges);mx,my=project_local(X,Y)
    sx,sy=project_local(sites.x,sites.y)
    gx=np.rint(xedges[:-1]/cell).astype(int);gy=np.rint(yedges[:-1]/cell).astype(int)
    index=pd.MultiIndex.from_product([gy,gx],names=['gy','gx'])
    panels=[]
    for block in range(4):
        data=grid.loc[grid.block==block].set_index(['gy','gx']).reindex(index)
        mask=(data.activity.to_numpy().reshape(len(gy),len(gx))<minimum)|data.activity.isna().to_numpy().reshape(len(gy),len(gx))
        panels.append((block,data,mask))
    maximum=max(float(data.loc[(data.activity>=minimum),'events_per_hour'].max())
        for _,data,_ in panels)
    for name,measure,cmap,norm,label in [
        ('activity','events_per_hour','YlOrRd',Normalize(vmin=0,vmax=maximum),
            'Arrivals + departures per hour'),
        ('balances','ratio','coolwarm',TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),
            'Normalised arrival–departure balance')]:
        fig,axs=plt.subplots(2,2,figsize=(9,6.5),layout='constrained')
        for i,(ax,(block,data,mask)) in enumerate(zip(axs.flat,panels)):
            values=data[measure].to_numpy().reshape(len(gy),len(gx))
            values=np.ma.masked_where(mask|~np.isfinite(values),values)
            ax.imshow(context,extent=extent,origin='upper',zorder=0)
            mesh=ax.pcolormesh(mx,my,values,norm=norm,cmap=cmap,alpha=.8,
                rasterized=True,zorder=1)
            ax.scatter(sx,sy,marker='+',s=24,color='#163e63',linewidths=1,zorder=3)
            if name=='activity' and i==0:
                offsets={'Ercolani':(6,-11),'Manzi':(-6,9),'Parri':(6,9),
                    'Sabotino':(-6,-10),'San Donato':(6,6),'Sturzo':(-6,-11),
                    'Zanardi':(-6,-11)}
                for position,site in enumerate(sites.index):
                    offset=offsets.get(site,(6,6))
                    ax.annotate(site.replace('_',' '),xy=(sx[position],sy[position]),
                        xytext=offset,textcoords='offset points',fontsize=8,
                        ha='left' if offset[0]>0 else 'right',color='#163e63',
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1},
                        arrowprops={'arrowstyle':'-','color':'#163e63','lw':.5},zorder=5)
            ax.set(xlim=(mx.min(),mx.max()),ylim=(my.min(),my.max()),aspect='equal',
                title=f'({chr(97+i)}) {MAP_LABELS[block]} · 2023–2024')
            ax.set_xticks([]);ax.set_yticks([])
            ax.spines[['top','right']].set_visible(True)
            bx,by=project_local(np.array([-6.3,-4.3]),np.array([-4.45,-4.45]))
            ax.plot(bx,by,color='black',lw=2,zorder=4)
            ax.text(bx.mean(),by[0]+180,'2 km',ha='center',fontsize=7,zorder=4)
        colorbar=fig.colorbar(mesh,ax=axs.ravel().tolist(),shrink=.82,label=label)
        if measure=='ratio':colorbar.set_ticks([-1,-.5,0,.5,1])
        fig.supxlabel(ATTRIBUTION+'\nCrosses: counters · Same valid periods and 500 m cells in both figures',fontsize=6)
        fig.savefig(TEXT/'figures'/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(TEXT/'figures'/f'{name}.png',bbox_inches='tight',dpi=220)
        plt.close(fig)
    details=[]
    for block,data,mask in panels:
        selected=grid.loc[grid.block==block]
        p=periods.loc[periods.block==block]
        details.append({'block':block,'label':MAP_LABELS[block],
            'hours':int(p.hours.sum()),'periods':len(p),
            'departures':int(selected.departures.sum()),'arrivals':int(selected.arrivals.sum()),
            'displayed_cells':int((~mask).sum())})
    dump('sharing-map-design.json',{'years':[2023,2024],'cell_km':cell,
        'display_minimum_activity':minimum,'local_extent_km':[-7,8,-5,6],
        'activity':'(arrivals + departures) / valid period hours',
        'balance':'(arrivals - departures) / (arrivals + departures)',
        'activity_scale':[0,maximum],'balance_scale':[-1,0,1],'panels':details,
        'counter_sites':sorted(sites.index),
        'period_selection':'Complete 06–10, 10–15, 15–20, 20–06 windows; all constituent hours must be within 2023–2024 and have usable sharing coverage. Same selection for both maps.',
        'date_convention':'Service date starts at 06:00; overnight 20–06 belongs to the evening date.',
        'event_assignment':'Each departure uses its start time; each arrival its return time.'})

def leading_od_links(od,year,regime,limit=40):
    """Top inter-cell links; shares use all retained trips in the panel."""
    panel=od.loc[(od.year==year)&(od.regime==regime)]
    total=int(panel.n.sum())
    links=panel.loc[((panel.ogx!=panel.dgx)|(panel.ogy!=panel.dgy))&(panel.n>=10)]
    links=links.sort_values(['n','ogx','ogy','dgx','dgy'],
        ascending=[False,True,True,True,True]).head(limit).copy()
    links['share']=links.n/total if total else 0.
    return links,total

def draw_od_map(od,sites):
    """Four comparable year/day-night panels; lines are not routed paths."""
    bounds_local=(-7,8,-5,6)
    west=LON0+bounds_local[0]/(111.195*np.cos(np.deg2rad(LAT0)))
    east=LON0+bounds_local[1]/(111.195*np.cos(np.deg2rad(LAT0)))
    south=LAT0+bounds_local[2]/111.195
    north=LAT0+bounds_local[3]/111.195
    context,extent=basemap((west,south,east,north))
    left,bottom=project_local(bounds_local[0],bounds_local[2])
    right,top=project_local(bounds_local[1],bounds_local[3])
    panels=[(year,regime,*leading_od_links(od,year,regime))
        for year in [2023,2024] for regime in ['day','night']]
    max_share=max((links.share.max() for _,_,links,_ in panels if len(links)),default=1.)
    def width(share):
        return .35+3.2*np.asarray(share)/max_share
    fig,axs=plt.subplots(2,2,figsize=(9,7.2),layout='constrained')
    sx,sy=project_local(sites.x,sites.y)
    details=[]
    for index,(ax,(year,regime,links,total)) in enumerate(zip(axs.flat,panels)):
        ax.imshow(context,extent=extent,origin='upper',zorder=0)
        ox,oy=project_local(links.ogx+.5,links.ogy+.5)
        dx,dy=project_local(links.dgx+.5,links.dgy+.5)
        segments=np.stack([np.column_stack([ox,oy]),np.column_stack([dx,dy])],axis=1)
        ax.add_collection(LineCollection(segments,linewidths=width(links.share),
            colors='#0072B2',alpha=.7,zorder=2))
        ax.scatter(sx,sy,c='#163e63',marker='+',s=24,linewidths=1,zorder=3)
        label='Day (06–23)' if regime=='day' else 'Night (23–06)'
        ax.set(xlim=(left,right),ylim=(bottom,top),aspect='equal',
            title=f'({chr(97+index)}) {year} · {label}')
        ax.set_xticks([]);ax.set_yticks([])
        bx,by=project_local(np.array([-6.3,-4.3]),np.array([-4.45,-4.45]))
        ax.plot(bx,by,color='black',lw=2,zorder=4)
        ax.text(bx.mean(),by[0]+180,'2 km',ha='center',fontsize=7,zorder=4)
        details.append({'year':year,'regime':regime,'retained_trips':total,
            'displayed_links':len(links),'displayed_trips':int(links.n.sum())})
    shares=[max_share*f for f in [.25,.5,1.]]
    handles=[Line2D([],[],color='#0072B2',lw=float(width(s)),label=f'{100*s:.2f}%')
        for s in shares]
    axs[0,0].legend(handles=handles,title='Share of panel trips',loc='upper left',
        fontsize=6,title_fontsize=6,framealpha=.9)
    fig.supxlabel(ATTRIBUTION+'\nBlue lines: OD links, not routes · Crosses: counters · Common line-width scale',fontsize=6)
    fig.savefig(TEXT/'figures/od-links.pdf',bbox_inches='tight')
    fig.savefig(TEXT/'figures/od-links.png',bbox_inches='tight',dpi=220)
    plt.close(fig)
    dump('city-od-map.json',{'cell_km':1,'links_per_panel':40,
        'minimum_link_count':10,'local_extent_km':list(bounds_local),
        'line_width':'Common scale: OD count / all retained trips in the year/period panel',
        'maximum_displayed_share':float(max_share),'panels':details,
        'counter_sites':sorted(sites.index),
        'note':'Straight cell-centre connectors, not street routes. 2023 coverage is incomplete after filtering.'})

def six_hour_balances(t,year=2024,cell=.5):
    """Count each endpoint at its own event time, including calendar-year crossings."""
    counts=[]
    for prefix,time,kind in [('o','start','departures'),('d','end','arrivals')]:
        selected=t.loc[t[time].dt.year==year]
        events=pd.DataFrame({'block':six_hour_blocks(selected[time]),
            'gx':np.floor(selected[prefix+'x']/cell).astype('int16'),
            'gy':np.floor(selected[prefix+'y']/cell).astype('int16')})
        counts.append(events.groupby(['block','gx','gy']).size().rename(kind))
    result=pd.concat(counts,axis=1).fillna(0).astype('int64').reset_index()
    result['activity']=result.arrivals+result.departures
    result['balance']=result.arrivals-result.departures
    result['ratio']=result.balance/result.activity
    result['block_label']=result.block.map(dict(enumerate(BLOCK_LABELS)))
    result['cell_km']=cell;result['year']=year
    return result

def draw_balance_map(t,sites):
    year=2024;cell=.5;minimum=200
    # Shared cartography and fixed cells; activity is arrivals + departures.
    xedges=np.arange(-7,8+cell,cell);yedges=np.arange(-5,6+cell,cell)
    balances=six_hour_balances(t,year,cell)
    balances.to_csv(OUT/'grid-balances.csv',index=False)
    west=LON0+xedges[0]/(111.195*np.cos(np.deg2rad(LAT0)))
    east=LON0+xedges[-1]/(111.195*np.cos(np.deg2rad(LAT0)))
    south=LAT0+yedges[0]/111.195;north=LAT0+yedges[-1]/111.195
    context,extent=basemap((west,south,east,north))
    X,Y=np.meshgrid(xedges,yedges);mx,my=project_local(X,Y)
    norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    fig,axs=plt.subplots(2,2,figsize=(9,7.2),layout='constrained')
    sx,sy=project_local(sites.x,sites.y)
    gx=np.rint(xedges[:-1]/cell).astype(int);gy=np.rint(yedges[:-1]/cell).astype(int)
    activity_by_block=[]
    for block,ax in enumerate(axs.flat):
        b=balances.loc[balances.block==block].set_index(['gy','gx'])
        index=pd.MultiIndex.from_product([gy,gx],names=['gy','gx'])
        b=b.reindex(index)
        values=b.ratio.to_numpy().reshape(len(gy),len(gx))
        activity=b.activity.to_numpy().reshape(len(gy),len(gx))
        values=np.ma.masked_where((activity<minimum)|~np.isfinite(values),values)
        ax.imshow(context,extent=extent,origin='upper',zorder=0)
        mesh=ax.pcolormesh(mx,my,values,norm=norm,cmap='coolwarm',alpha=.76,rasterized=True,zorder=1)
        ax.scatter(sx,sy,marker='+',s=24,color='#163e63',linewidths=1,zorder=3)
        ax.set(xlim=(mx.min(),mx.max()),ylim=(my.min(),my.max()),aspect='equal',
            title=f'({chr(97+block)}) {BLOCK_LABELS[block]} · {year}')
        ax.set_xticks([]);ax.set_yticks([])
        # Approximate local ground-distance bar, transformed into map coordinates.
        bx,by=project_local(np.array([-6.3,-4.3]),np.array([-4.45,-4.45]))
        ax.plot(bx,by,color='black',lw=2,zorder=4)
        ax.text(bx.mean(),by[0]+180,'2 km',ha='center',fontsize=7,zorder=4)
        activity_by_block.append({'block':block,'label':BLOCK_LABELS[block],
            'departures':int(balances.loc[balances.block==block,'departures'].sum()),
            'arrivals':int(balances.loc[balances.block==block,'arrivals'].sum()),
            'displayed_cells':int(values.count())})
    fig.colorbar(mesh,ax=axs.ravel().tolist(),shrink=.8,ticks=[-1,-.5,0,.5,1],
        label='(Arrivals − departures) / activity')
    fig.supxlabel(ATTRIBUTION+'\nBlue crosses: retained counter sections',fontsize=6)
    fig.savefig(TEXT/'figures/balances.pdf',bbox_inches='tight')
    fig.savefig(TEXT/'figures/balances.png',bbox_inches='tight',dpi=220)
    plt.close(fig)
    dump('city-balance-map.json',{'year':year,'cell_km':cell,'display_minimum_activity':minimum,
        'local_extent_km':[-7,8,-5,6],'colour_scale':{'minimum':-1,'centre':0,'maximum':1},
        'activity_definition':'arrivals + departures',
        'departure_time':'recorded rental start, local calendar time',
        'arrival_time':'recorded rental end, local calendar time',
        'interval_convention':'left-closed, right-open; 18:00–24:00 is 18:00–00:00 next day',
        'blocks':activity_by_block,'counter_sites':sorted(sites.index),
        'note':'Rental durations >=30 seconds; excluded counter sites do not remove rental endpoints.'})
