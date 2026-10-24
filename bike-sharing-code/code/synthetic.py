"""Known-demand network experiment, explicitly NOT a reconstruction of Bologna."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,TEXT,SEED,dump

def network(seed=SEED):
    rng=np.random.default_rng(seed)
    g=nx.grid_2d_graph(5,5)
    for u,v in g.edges:
        # Fixed edge heterogeneity; an alternative network favours green corridors.
        g[u][v]['short']=1+rng.uniform(0,.08)
        g[u][v]['green']=g[u][v]['short']*(.65 if u[0]==2 or v[0]==2 else 1.35)
    nodes=list(g.nodes);pairs=[(u,v) for u in nodes for v in nodes if u!=v]
    edges=list(g.edges);lookup={}
    for i,(u,v) in enumerate(edges):lookup[u,v]=2*i;lookup[v,u]=2*i+1
    mats=[]
    for weight in ['short','green']:
        A=np.zeros((2*len(edges),len(pairs)))
        paths=dict(nx.all_pairs_dijkstra_path(g,weight=weight))
        for j,(o,d) in enumerate(pairs):
            path=paths[o][d]
            for u,v in zip(path[:-1],path[1:]):A[lookup[u,v],j]=1
        mats.append(A)
    return g,nodes,pairs,edges,mats

def expand(p,A,y):
    pred=A@p
    return p*max(y.sum()/max(pred.sum(),1e-12),1e-8)

def fit(p,A,y,tau=.2):
    """Poisson negative log likelihood + tau*KL(q||p), bounded q>0."""
    p=np.maximum(p,1e-6)
    def fun(q):
        mu=np.maximum(A@q,1e-10)
        val=np.sum(mu-y*np.log(mu))+tau*np.sum(q*np.log(q/p)-q+p)
        grad=A.T@(1-y/mu)+tau*np.log(q/p)
        return val,grad
    r=minimize(fun,p.copy(),jac=True,method='L-BFGS-B',bounds=[(1e-6,None)]*len(p),
               options={'maxiter':1000,'ftol':1e-10,'gtol':1e-5,'maxls':40})
    return r.x, bool(r.success),float(np.max(np.abs(r.jac)))

def aggregate(A,indices,directional):
    z=A[np.array([[2*i,2*i+1] for i in indices]).ravel()]
    return z if directional else z.reshape(len(indices),2,-1).sum(axis=1)

def evaluate(q,gravity,sharing,yfull,truth_A,short,pool,test,rep,bias_strength):
    records=[]
    for routing,A in [('correct',truth_A),('shortest_only',short)]:
        for directional in [True,False]:
            for budget in [4,8,10]:
                train=pool[:budget];Ac=aggregate(A,train,directional)
                # Identical directional test target under both calibration designs.
                At=aggregate(A,test,True)
                yc=yfull[np.array([[2*i,2*i+1] for i in train]).ravel()]
                yt=yfull[np.array([[2*i,2*i+1] for i in test]).ravel()]
                if not directional:yc=yc.reshape(-1,2).sum(axis=1)
                ps=expand(sharing,Ac,yc);pg=expand(gravity,Ac,yc)
                for model,p in [('uniform_expansion',ps),('sharing_prior',ps),('gravity_prior',pg)]:
                    if model=='uniform_expansion':qh,ok,grad=p,True,0.
                    else:qh,ok,grad=fit(p,Ac,yc)
                    records.append({'replicate':rep,'bias_strength':bias_strength,'routing':routing,'directional':directional,'sections':budget,'model':model,
                        'od_relative_l1':float(np.sum(abs(qh-q))/q.sum()),
                        'test_wape':float(np.sum(abs(At@qh-yt))/yt.sum()),
                        'train_wape':float(np.sum(abs(Ac@qh-yc))/yc.sum()),
                        'converged':ok,'max_gradient':grad,'rank':int(np.linalg.matrix_rank(Ac)),
                        'uncovered_od_fraction':float((Ac.sum(axis=0)==0).mean()),
                        'uncovered_demand_fraction':float(q[Ac.sum(axis=0)==0].sum()/q.sum())})
    return records

def main(reps=20):
    g,nodes,pairs,edges,(short,green)=network()
    truth_A=.7*short+.3*green
    # 12 physical sections, independently selected once, fixed across all replicates.
    selection=np.random.default_rng(SEED+1).choice(len(edges),12,replace=False)
    test=selection[-2:];pool=selection[:-2]
    records=[];regularization=[]
    for rep in range(reps):
        rng=np.random.default_rng(SEED+rep+100)
        attractiveness=np.exp(rng.normal(0,.4,len(nodes)));pop=dict(zip(nodes,attractiveness))
        dist=np.array([abs(o[0]-d[0])+abs(o[1]-d[1]) for o,d in pairs])
        gravity=np.array([pop[o]*pop[d] for o,d in pairs])*np.exp(-dist/2)
        directional_pattern=np.array([np.exp(.35*(d[0]-o[0])+.3*(d[1]-o[1])) for o,d in pairs])
        q=gravity*directional_pattern*np.exp(rng.normal(0,.2,len(pairs)));q=q/q.sum()*50000
        selection_bias=np.array([np.exp(.65*(o[0]-2)-.35*(d[1]-2)) for o,d in pairs])*np.exp(-.2*dist)
        yfull=rng.poisson(truth_A@q)
        for bias_strength in [0,1]:
            sample=rng.poisson(q*.06*selection_bias**bias_strength)
            sharing=sample+.5  # Positive support includes OD pairs unobserved in sharing.
            records.extend(evaluate(q,gravity,sharing,yfull,truth_A,short,pool,test,rep,bias_strength))
            Ac=aggregate(truth_A,pool,True);At=aggregate(truth_A,test,True)
            yc=yfull[np.array([[2*i,2*i+1] for i in pool]).ravel()]
            yt=yfull[np.array([[2*i,2*i+1] for i in test]).ravel()]
            for model,p in [('sharing_prior',sharing),('gravity_prior',gravity)]:
                p=expand(p,Ac,yc)
                for tau in [.02,.2,2.]:
                    qh,ok,_=fit(p,Ac,yc,tau=tau)
                    regularization.append({'replicate':rep,'bias_strength':bias_strength,'model':model,'tau':tau,
                        'od_relative_l1':float(abs(qh-q).sum()/q.sum()),'test_wape':float(abs(At@qh-yt).sum()/yt.sum()),'converged':ok})
        print('Synthetic replicate',rep+1,'/',reps,flush=True)
    df=pd.DataFrame(records);df.to_csv(OUT/'synthetic-replicates.csv',index=False)
    pd.DataFrame(regularization).to_csv(OUT/'regularization-sensitivity.csv',index=False)
    summary=df.groupby(['bias_strength','routing','directional','sections','model'])[['od_relative_l1','test_wape','train_wape','rank','uncovered_od_fraction','uncovered_demand_fraction']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(x).strip('_') if isinstance(x,tuple) else x for x in summary.columns]
    summary.to_csv(OUT/'synthetic-summary.csv',index=False)
    # Constructively show observational equivalence at ALL 12 selected sections.
    Af=aggregate(truth_A,selection,True)
    _,group=np.unique(np.round(Af,10).T,axis=0,return_inverse=True)
    equivalent=np.zeros_like(q)
    for k in np.unique(group):
        ids=np.where(group==k)[0];equivalent[ids[0]]=q[ids].sum()
    witness={'od_pairs':len(pairs),'physical_sections':12,'directional_rows':len(Af),
             'rank_directional':int(np.linalg.matrix_rank(Af)),
             'rank_aggregated':int(np.linalg.matrix_rank(aggregate(truth_A,selection,False))),
             'distinct_column_signatures':len(np.unique(group)),
             'zero_columns':int((Af.sum(axis=0)==0).sum()),
             'equivalent_od_relative_l1':float(abs(q-equivalent).sum()/q.sum()),
             'equivalent_count_max_difference':float(abs(Af@q-Af@equivalent).max()),
             'replicates':reps,'fits_failed':int((~df.converged).sum()),'tau':.2,'bias_strengths':[0,1],
             'heldout_sections':[str(edges[i]) for i in test],
             'all_selected_sections':[str(edges[i]) for i in selection]}
    dump('synthetic-design.json',witness)
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.8))
    for model,color in [('uniform_expansion','#999999'),('sharing_prior','#0072B2'),('gravity_prior','#D55E00')]:
        for direct,ls in [(True,'-'),(False,'--')]:
            z=df.loc[(df.bias_strength==1)&(df.routing=='correct')&(df.directional==direct)&(df.model==model)]
            for ax,col in zip(axs,['test_wape','od_relative_l1']):
                a=z.groupby('sections')[col].agg(['mean','std'])
                ax.plot(a.index,a['mean'],ls,color=color,marker='.',label=model.replace('_',' ')+(' (dir.)' if direct else ' (sum)'))
                if direct:ax.fill_between(a.index,a['mean']-a['std'],a['mean']+a['std'],color=color,alpha=.08)
                ax.set(xlabel='Calibration sections',ylabel='Held-out count WAPE' if col=='test_wape' else 'OD relative L1 error',xticks=[4,8,10])
    axs[1].legend(fontsize=5.5);fig.tight_layout();fig.savefig(TEXT/'figures/synthetic.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.4,2.7));pos={n:n for n in nodes}
    nx.draw_networkx_edges(g,pos,ax=ax,edge_color='#cccccc')
    nx.draw_networkx_nodes(g,pos,ax=ax,node_size=15,node_color='gray')
    nx.draw_networkx_edges(g,pos,edgelist=[edges[i] for i in pool],ax=ax,edge_color='#0072B2',width=2)
    nx.draw_networkx_edges(g,pos,edgelist=[edges[i] for i in test],ax=ax,edge_color='#D55E00',width=3)
    ax.set_title('Synthetic grid: blue calibration, orange test');ax.axis('off');fig.savefig(TEXT/'figures/synthetic-network.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
