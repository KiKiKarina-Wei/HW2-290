import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

xl        = pd.read_excel('Prices.xlsx', sheet_name=None)
df_elec   = xl['PRICE_ELECTRIC'].copy(); df_elec.columns = ['date','hour','price_e']
df_gas    = xl['PRICE_GAS'].copy();      df_gas.columns  = ['date','price_g']
df_gas['date']  = pd.to_datetime(df_gas['date']).dt.date
df_elec['date'] = pd.to_datetime(df_elec['date']).dt.date
df = df_elec.merge(df_gas, on='date').sort_values(['date','hour']).reset_index(drop=True)
T = len(df)   # 168 hours

price_e   = df['price_e'].values
price_g   = df['price_g'].values

CO2_PRICE    = 28.3      # $/tonne  (from dataset)
CO2_FACTOR   = 0.05306   # tonne CO2/MMBtu  (EIA standard for natural gas)
fuel_rate    = price_g + CO2_FACTOR * CO2_PRICE   # $/MMBtu all-in at each hour

DAY_LABELS   = ['Mar 21','Mar 22','Mar 23','Mar 24','Mar 25','Mar 26','Mar 27']
HOURS        = np.arange(1, T+1)


# TASK 2

print("=" * 60)
print("TASK 2: CAISO Multi-Stage Generator Optimization")
print("=" * 60)

CONFIGS = [1, 2, 3, 4, 5]   # 1=off, 2=CT1 SC, 3=CT1+CT2 SC, 4=1x1, 5=2x1
ACTIVE  = [2, 3, 4, 5]

cap  = {1:0,   2:190,  3:380,  4:340,  5:610}
pmin = {1:0,   2:57,   3:114,  4:150,  5:312}
MUT  = {1:0,   2:1,    3:1,    4:2,    5:3}
MDT  = {1:0,   2:1,    3:1,    4:2,    5:3}
VOM  = {1:0.0, 2:5.00, 3:5.00, 4:2.50, 5:2.00}

# Startup costs from off (non-fuel $/start, fuel MMBtu/start)
# Config 3: both CTs start independently = 2 × CT startup
# Config 5: cumulative from off = 1x1 + CT2 startup
SU_D = {1:0, 2:7250,  3:14500, 4:16500, 5:23750}
SU_F = {1:0, 2:220,   3:440,   4:850,   5:1070}
# Incremental cost for 4→5 transition (CT2 startup only)
SU_D_45 = 7250;  SU_F_45 = 220

IHR_CT  = [8.692, 9.177, 9.564]
IHR_1x1 = [6.421, 6.525, 6.640]
IHR_2x1 = [6.292, 6.361, 6.452]
AHR_MIN = {1:0, 2:12.591, 3:12.591, 4:7.695, 5:7.121}

def build_segs(pmin_mw, cap_mw, ihr):
    pts = [pmin_mw, 0.6*cap_mw, 0.8*cap_mw, cap_mw]
    return [(pts[i+1]-pts[i], ihr[i]) for i in range(3)]

SEG = {1:[], 2:build_segs(pmin[2],cap[2],IHR_CT),
             3:build_segs(pmin[3],cap[3],IHR_CT),
             4:build_segs(pmin[4],cap[4],IHR_1x1),
             5:build_segs(pmin[5],cap[5],IHR_2x1)}
NS = {c: len(SEG[c]) for c in CONFIGS}

#Build MILP
prob = LpProblem("CCGT_CAISO", LpMaximize)

x = {(c,t): LpVariable(f"x_{c}_{t}", cat='Binary')  for c in CONFIGS for t in range(T)}
y = {(c,t): LpVariable(f"y_{c}_{t}", cat='Binary')  for c in CONFIGS for t in range(T)}
z = {(c,t): LpVariable(f"z_{c}_{t}", cat='Binary')  for c in CONFIGS for t in range(T)}
p = {(c,s,t): LpVariable(f"p_{c}_{s}_{t}", lowBound=0, upBound=SEG[c][s][0])
     for c in ACTIVE for s in range(NS[c]) for t in range(T)}

# Objective
rev    = lpSum(price_e[t]*(pmin[c]*x[c,t] + lpSum(p[c,s,t] for s in range(NS[c])))
               for c in ACTIVE for t in range(T))
fuel_c = lpSum(fuel_rate[t]*(AHR_MIN[c]*pmin[c]*x[c,t] +
               lpSum(SEG[c][s][1]*p[c,s,t] for s in range(NS[c])))
               for c in ACTIVE for t in range(T))
vom_c  = lpSum(VOM[c]*(pmin[c]*x[c,t] + lpSum(p[c,s,t] for s in range(NS[c])))
               for c in ACTIVE for t in range(T))
su_c   = (lpSum((SU_D[c]+SU_F[c]*fuel_rate[t])*y[c,t] for c in [2,3,4] for t in range(T)) +
          lpSum((SU_D_45+SU_F_45*fuel_rate[t])*y[5,t] for t in range(T)))
prob  += rev - fuel_c - vom_c - su_c

# Constraints
for t in range(T):
    prob += lpSum(x[c,t] for c in CONFIGS) == 1, f"one_cfg_{t}"
for c in ACTIVE:
    for s in range(NS[c]):
        for t in range(T):
            prob += p[c,s,t] <= SEG[c][s][0]*x[c,t], f"seg_{c}_{s}_{t}"
for c in ACTIVE:
    for t in range(T):
        if t == 0:
            prob += y[c,0] == x[c,0];  prob += z[c,0] == 0
        else:
            prob += y[c,t] >= x[c,t]-x[c,t-1];  prob += y[c,t] <= x[c,t]
            prob += y[c,t] <= 1-x[c,t-1]
            prob += z[c,t] >= x[c,t-1]-x[c,t];  prob += z[c,t] <= x[c,t-1]
            prob += z[c,t] <= 1-x[c,t]
    for t in range(T):
        if MUT[c] > 1:
            for tau in range(t+1, min(t+MUT[c], T)):
                prob += x[c,tau] >= y[c,t]
        if MDT[c] > 1:
            for tau in range(t+1, min(t+MDT[c], T)):
                prob += 1-x[c,tau] >= z[c,t]
for t in range(1, T):
    prob += y[5,t] <= x[4,t-1]   # 4→5 transition constraint

prob.solve(PULP_CBC_CMD(msg=0, timeLimit=300))
print(f"Status: {LpStatus[prob.status]}")

# Extract results
rows2 = []
for t in range(T):
    cfg = next(c for c in CONFIGS if value(x[c,t]) > 0.5)
    mw_seg = sum(value(p[cfg,s,t]) or 0 for s in range(NS[cfg])) if cfg in ACTIVE else 0
    mw = (pmin[cfg] if cfg in ACTIVE else 0) + mw_seg
    rows2.append({'OPERATING_DATE': df['date'].iloc[t], 'HOUR_ENDING': df['hour'].iloc[t],
                  'PRICE_ELECTRIC': price_e[t], 'PRICE_GAS': price_g[t],
                  'CONFIGURATION_ACTIVE': cfg, 'MW_GENERATION': round(mw,2)})
df2 = pd.DataFrame(rows2)

def running_fuel_t2(row):
    c, mw, t = row['CONFIGURATION_ACTIVE'], row['MW_GENERATION'], row.name
    if c == 1 or mw == 0: return 0.0
    f = AHR_MIN[c]*pmin[c]; rem = mw - pmin[c]
    for bk,ihr in SEG[c]:
        chunk=min(rem,bk); f+=ihr*chunk; rem-=chunk
        if rem<=1e-6: break
    return f*fuel_rate[t]

df2['revenue']   = df2['PRICE_ELECTRIC']*df2['MW_GENERATION']
df2['fuel_cost'] = df2.apply(running_fuel_t2, axis=1)
df2['vom_cost']  = df2.apply(lambda r: VOM[r['CONFIGURATION_ACTIVE']]*r['MW_GENERATION'], axis=1)
df2['startup_cost'] = 0.0
for t in range(T):
    c = df2['CONFIGURATION_ACTIVE'].iloc[t]
    if c == 1: continue
    c_prev = df2['CONFIGURATION_ACTIVE'].iloc[t-1] if t>0 else 1
    if c != c_prev:
        df2.loc[t,'startup_cost'] = (SU_D_45+SU_F_45*fuel_rate[t] if (c==5 and c_prev==4)
                                     else SU_D[c]+SU_F[c]*fuel_rate[t])

t2_rev = df2['revenue'].sum();  t2_fuel = df2['fuel_cost'].sum()
t2_vom = df2['vom_cost'].sum(); t2_su   = df2['startup_cost'].sum()
t2_gm  = t2_rev - t2_fuel - t2_vom - t2_su
t2_cf  = df2['MW_GENERATION'].mean()/cap[5]*100
t2_ns  = (df2['startup_cost']>0).sum()

print(f"  Gross Margin:    ${t2_gm:,.0f}")
print(f"  GM ($/kW):       ${t2_gm/cap[5]/1000:.4f}")
print(f"  Capacity Factor: {t2_cf:.1f}%")
print(f"  Starts:          {t2_ns}")

csv2_cols = ['OPERATING_DATE','HOUR_ENDING','PRICE_ELECTRIC','PRICE_GAS',
             'CONFIGURATION_ACTIVE','MW_GENERATION']
df2[csv2_cols].to_csv('CCGT_CAISO.csv', index=False)

#Plots
COLORS   = {1:'#e9ecef',2:'#f4a261',3:'#e76f51',4:'#2a9d8f',5:'#264653'}
CFG_NAMES= {1:'1: Off',2:'2: CT1 SC',3:'3: CT1+CT2',4:'4: 1×1',5:'5: 2×1'}
fig, axes = plt.subplots(2,1,figsize=(15,9),facecolor='white')
cfg_arr = df2['CONFIGURATION_ACTIVE'].values; mw_arr = df2['MW_GENERATION'].values
bar_c   = [COLORS[c] for c in cfg_arr]

ax = axes[0]
ax.bar(HOURS, mw_arr, color=bar_c, edgecolor='none', width=1.0)
ax.axhline(cap[5],  color='#e63946', lw=1.2, ls='--', alpha=0.7)
ax.axhline(pmin[5], color='#457b9d', lw=1.0, ls=':',  alpha=0.6)
for d in range(1,7): ax.axvline(d*24+0.5, color='gray', lw=0.4, alpha=0.4)
for d,lbl in enumerate(DAY_LABELS): ax.text(d*24+12, 630, lbl, ha='center', fontsize=8.5, color='#555')
ax.set_xlim(0.5,T+0.5); ax.set_ylim(0,680)
ax.set_ylabel('MW Generation', fontsize=11); ax.set_xlabel('Hour', fontsize=10)
ax.set_title('Task 2 CAISO — MW Generation (March 21–27, 2022)', fontsize=12, fontweight='bold')
patches=[mpatches.Patch(color=COLORS[c],label=CFG_NAMES[c]) for c in CONFIGS]
ax.legend(handles=patches, loc='upper left', fontsize=8.5, ncol=2, framealpha=0.85)
ax.grid(axis='y', alpha=0.25)

ax2 = axes[1]
ax2.bar(HOURS, cfg_arr, color=bar_c, edgecolor='none', width=1.0)
ax2.set_yticks([1,2,3,4,5]); ax2.set_yticklabels([CFG_NAMES[c] for c in CONFIGS], fontsize=9)
ax2.set_ylabel('Active Configuration', fontsize=11); ax2.set_xlabel('Hour', fontsize=10)
ax2.set_title('Task 2 CAISO — Configuration Mode (March 21–27, 2022)', fontsize=12, fontweight='bold')
ax2.set_xlim(0.5,T+0.5); ax2.set_ylim(0.5,5.8)
for d in range(1,7): ax2.axvline(d*24+0.5, color='gray', lw=0.4, alpha=0.4)
for d,lbl in enumerate(DAY_LABELS): ax2.text(d*24+12, 5.55, lbl, ha='center', fontsize=8.5, color='#555')
ax2.grid(axis='y', alpha=0.2)
plt.tight_layout(pad=2.0)
plt.savefig('ccgt_dispatch_plots.png', dpi=150, bbox_inches='tight'); plt.close()


# TASK 3

print("\n" + "=" * 60)
print("TASK 3: PJM Pseudo-Unit Optimization")
print("=" * 60)

U1 = dict(cap=340, pmin=150, mut=2, mdt=2, su_d=16500, su_f=850,
          ahr_min=7.695, vom=2.50,
          segs=[(54,6.421),(68,6.525),(68,6.640)])
U2 = dict(cap=270, pmin=162, mut=3, mdt=3, su_d=7250,  su_f=220,
          ahr_min=6.590, vom=1.370,
          segs=[(54,6.026),(54,6.215)])
UNITS = {1:U1, 2:U2}

prob3 = LpProblem("CCGT_Pseudo", LpMaximize)
x3 = {(u,t): LpVariable(f"x3_{u}_{t}", cat='Binary') for u in [1,2] for t in range(T)}
y3 = {(u,t): LpVariable(f"y3_{u}_{t}", cat='Binary') for u in [1,2] for t in range(T)}
z3 = {(u,t): LpVariable(f"z3_{u}_{t}", cat='Binary') for u in [1,2] for t in range(T)}
p3 = {(u,s,t): LpVariable(f"p3_{u}_{s}_{t}", lowBound=0,
                           upBound=UNITS[u]['segs'][s][0])
      for u in [1,2] for s in range(len(UNITS[u]['segs'])) for t in range(T)}

def umw(u,t):
    return UNITS[u]['pmin']*x3[u,t]+lpSum(p3[u,s,t] for s in range(len(UNITS[u]['segs'])))
def ufuel(u,t):
    return (UNITS[u]['ahr_min']*UNITS[u]['pmin']*x3[u,t]+
            lpSum(UNITS[u]['segs'][s][1]*p3[u,s,t] for s in range(len(UNITS[u]['segs']))))

prob3 += (lpSum(price_e[t]*(umw(1,t)+umw(2,t)) for t in range(T))
         -lpSum(fuel_rate[t]*(ufuel(1,t)+ufuel(2,t)) for t in range(T))
         -lpSum(UNITS[u]['vom']*umw(u,t) for u in [1,2] for t in range(T))
         -lpSum((UNITS[u]['su_d']+UNITS[u]['su_f']*fuel_rate[t])*y3[u,t]
                for u in [1,2] for t in range(T)))

for u in [1,2]:
    ns3=len(UNITS[u]['segs'])
    for t in range(T):
        for s in range(ns3):
            prob3 += p3[u,s,t] <= UNITS[u]['segs'][s][0]*x3[u,t]
        if t==0:
            prob3 += y3[u,0]==x3[u,0]; prob3 += z3[u,0]==0
        else:
            prob3 += y3[u,t] >= x3[u,t]-x3[u,t-1]; prob3 += y3[u,t] <= x3[u,t]
            prob3 += y3[u,t] <= 1-x3[u,t-1]
            prob3 += z3[u,t] >= x3[u,t-1]-x3[u,t]; prob3 += z3[u,t] <= x3[u,t-1]
            prob3 += z3[u,t] <= 1-x3[u,t]
        if UNITS[u]['mut']>1:
            for tau in range(t+1,min(t+UNITS[u]['mut'],T)):
                prob3 += x3[u,tau] >= y3[u,t]
        if UNITS[u]['mdt']>1:
            for tau in range(t+1,min(t+UNITS[u]['mdt'],T)):
                prob3 += 1-x3[u,tau] >= z3[u,t]
for t in range(T):
    prob3 += x3[2,t] <= x3[1,t]

prob3.solve(PULP_CBC_CMD(msg=0, timeLimit=300))
print(f"Status: {LpStatus[prob3.status]}")

rows3=[]
for t in range(T):
    def mw3(u):
        if value(x3[u,t])<0.5: return 0.0
        return round(UNITS[u]['pmin']+sum(value(p3[u,s,t]) or 0
               for s in range(len(UNITS[u]['segs']))),2)
    rows3.append({'OPERATING_DATE':df['date'].iloc[t],'HOUR_ENDING':df['hour'].iloc[t],
                  'PRICE_ELECTRIC':price_e[t],'PRICE_GAS':price_g[t],
                  'MW_GENERATION_Unit1':mw3(1),'MW_GENERATION_Unit2':mw3(2)})
df3=pd.DataFrame(rows3)
df3['MW_TOTAL']=df3['MW_GENERATION_Unit1']+df3['MW_GENERATION_Unit2']

def rfuel3(u,mw,t):
    if mw==0: return 0.0
    f=UNITS[u]['ahr_min']*UNITS[u]['pmin']; rem=mw-UNITS[u]['pmin']
    for bk,ihr in UNITS[u]['segs']:
        chunk=min(rem,bk); f+=ihr*chunk; rem-=chunk
        if rem<=1e-6: break
    return f*fuel_rate[t]

for u in [1,2]:
    cm=f'MW_GENERATION_Unit{u}'
    df3[f'rev_u{u}'] =df3['PRICE_ELECTRIC']*df3[cm]
    df3[f'fuel_u{u}']=df3.apply(lambda r,u=u,c=cm:rfuel3(u,r[c],r.name),axis=1)
    df3[f'vom_u{u}'] =UNITS[u]['vom']*df3[cm]
    df3[f'su_u{u}']  =0.0
for t in range(1,T):
    for u in [1,2]:
        cm=f'MW_GENERATION_Unit{u}'
        if df3[cm].iloc[t]>0 and df3[cm].iloc[t-1]==0:
            df3.loc[t,f'su_u{u}']=(UNITS[u]['su_d']+UNITS[u]['su_f']*fuel_rate[t])
for u in [1,2]:
    cm=f'MW_GENERATION_Unit{u}'
    if df3[cm].iloc[0]>0:
        df3.loc[0,f'su_u{u}']=(UNITS[u]['su_d']+UNITS[u]['su_f']*fuel_rate[0])

def t3_summary(u):
    r=df3[f'rev_u{u}'].sum(); f=df3[f'fuel_u{u}'].sum()
    v=df3[f'vom_u{u}'].sum(); s=df3[f'su_u{u}'].sum()
    gm=r-f-v-s; ns=(df3[f'su_u{u}']>0).sum()
    cf=df3[f'MW_GENERATION_Unit{u}'].mean()/UNITS[u]['cap']*100
    return dict(rev=r,fuel=f,vom=v,su=s,cost=f+v+s,gm=gm,ns=ns,cf=cf)
s1,s2=t3_summary(1),t3_summary(2)

print(f"  Unit 1 GM: ${s1['gm']:,.0f}  |  Unit 2 GM: ${s2['gm']:,.0f}")
print(f"  Combined GM: ${s1['gm']+s2['gm']:,.0f}")
print(f"  Capacity Factor: {s1['cf']:.1f}%")

df3[['OPERATING_DATE','HOUR_ENDING','PRICE_ELECTRIC','PRICE_GAS',
     'MW_GENERATION_Unit1','MW_GENERATION_Unit2']].to_csv('CCGT_PSEUDO.csv', index=False)

# plot
mw1=df3['MW_GENERATION_Unit1'].values; mw2=df3['MW_GENERATION_Unit2'].values
fig,axes=plt.subplots(2,1,figsize=(15,9),facecolor='white')
ax=axes[0]
ax.bar(HOURS,mw1,color='#2a9d8f',edgecolor='none',width=1.0,label='Unit 1 (1×1)')
ax.bar(HOURS,mw2,bottom=mw1,color='#264653',edgecolor='none',width=1.0,label='Unit 2 (CT2)')
ax.axhline(610,color='#e63946',lw=1.2,ls='--',alpha=0.7,label='Max 610 MW')
for d in range(1,7): ax.axvline(d*24+0.5,color='gray',lw=0.4,alpha=0.4)
for d,lbl in enumerate(DAY_LABELS): ax.text(d*24+12,630,lbl,ha='center',fontsize=8.5,color='#555')
ax.set_xlim(0.5,T+0.5);ax.set_ylim(0,680);ax.set_ylabel('MW Generation',fontsize=11)
ax.set_xlabel('Hour',fontsize=10)
ax.set_title('Task 3 Pseudo-Units — MW Generation (March 21–27, 2022)',fontsize=12,fontweight='bold')
ax.legend(loc='upper left',fontsize=9,ncol=2,framealpha=0.85);ax.grid(axis='y',alpha=0.25)
ax2=axes[1]
ax2.plot(HOURS,mw1,color='#2a9d8f',lw=1.5,label='Unit 1 (1×1 base)')
ax2.plot(HOURS,mw2,color='#264653',lw=1.5,label='Unit 2 (CT2 incr.)')
ax2.fill_between(HOURS,0,mw1,alpha=0.2,color='#2a9d8f')
ax2.fill_between(HOURS,0,mw2,alpha=0.2,color='#264653')
for d in range(1,7): ax2.axvline(d*24+0.5,color='gray',lw=0.4,alpha=0.4)
for d,lbl in enumerate(DAY_LABELS): ax2.text(d*24+12,295,lbl,ha='center',fontsize=8.5,color='#555')
ax2.set_xlim(0.5,T+0.5);ax2.set_ylim(0,310);ax2.set_ylabel('MW Generation',fontsize=11)
ax2.set_xlabel('Hour',fontsize=10)
ax2.set_title('Task 3 — Individual Pseudo-Unit Output (March 21–27, 2022)',fontsize=12,fontweight='bold')
ax2.legend(loc='upper left',fontsize=9,framealpha=0.85);ax2.grid(axis='y',alpha=0.25)
plt.tight_layout(pad=2.0)
plt.savefig('ccgt_pseudo_plots.png',dpi=150,bbox_inches='tight'); plt.close()

# TASK 4
print("\n" + "=" * 60)
print("TASK 4: Spark Spread Option")
print("=" * 60)

# 4a: k = 7
k7 = 7
payoff_k7    = (df['price_e'] - k7*fuel_rate).clip(lower=0)
gm_per_mw_k7 = payoff_k7.sum()
gm_per_kw_k7 = gm_per_mw_k7 / 1000
cf_k7        = (payoff_k7 > 0).mean() * 100

print(f"  k=7 Gross Margin ($/kW): ${gm_per_kw_k7:.4f}")
print(f"  k=7 Capacity Factor:     {cf_k7:.1f}%")

# 4b: best-fit k* for Task 2 and Task 3
CAP_PLANT = 610
k_vals  = np.linspace(1.0, 30.0, 20000)
gm_vals = np.array([(df['price_e'] - kk*fuel_rate).clip(lower=0).sum() for kk in k_vals])

t2_gm_val = t2_gm;  t3_gm_val = s1['gm']+s2['gm']
idx2_k = np.argmin(np.abs(gm_vals*CAP_PLANT - t2_gm_val))
idx3_k = np.argmin(np.abs(gm_vals*CAP_PLANT - t3_gm_val))
k_star2, k_star3 = k_vals[idx2_k], k_vals[idx3_k]

print(f"  k* matching Task 2 GM (${t2_gm_val:,.0f}): k* = {k_star2:.3f}")
print(f"  k* matching Task 3 GM (${t3_gm_val:,.0f}): k* = {k_star3:.3f}")


fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor='white')
k_plot=np.linspace(4,15,2000)
gm_plot=np.array([(df['price_e']-kk*fuel_rate).clip(lower=0).sum()*CAP_PLANT for kk in k_plot])
ax=axes[0]
ax.plot(k_plot,gm_plot,color='#264653',lw=2,label='GM(k)')
ax.axhline(t2_gm_val,color='#2a9d8f',ls='--',lw=1.3,label=f'Task 2 GM=${t2_gm_val/1000:.0f}k')
ax.axhline(t3_gm_val,color='#e76f51',ls='--',lw=1.3,label=f'Task 3 GM=${t3_gm_val/1000:.0f}k')
ax.axvline(k_star2,color='#2a9d8f',ls=':',lw=1); ax.axvline(k_star3,color='#e76f51',ls=':',lw=1)
ax.axvline(7,color='#f4a261',ls='-',lw=1.5,label='k=7 (Task 4a)')
ax.scatter([k_star2,k_star3,7],[gm_vals[idx2_k]*CAP_PLANT,gm_vals[idx3_k]*CAP_PLANT,gm_per_mw_k7*CAP_PLANT],
           color=['#2a9d8f','#e76f51','#f4a261'],zorder=5,s=60)
ax.annotate(f'k*={k_star2:.2f}',(k_star2,gm_vals[idx2_k]*CAP_PLANT),
            textcoords='offset points',xytext=(8,6),fontsize=9,color='#2a9d8f')
ax.annotate(f'k*={k_star3:.2f}',(k_star3,gm_vals[idx3_k]*CAP_PLANT),
            textcoords='offset points',xytext=(8,-16),fontsize=9,color='#e76f51')
ax.set_xlabel('Heat rate k (MMBtu/MWh)',fontsize=11);ax.set_ylabel('Gross Margin ($)',fontsize=11)
ax.set_title('Spark Spread GM vs k  (scaled to 610 MW)',fontsize=11,fontweight='bold')
ax.legend(fontsize=9,framealpha=0.85);ax.grid(alpha=0.25)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'${x/1000:.0f}k'))
ax2=axes[1]
ax2.bar(HOURS,payoff_k7.values,color='#f4a261',edgecolor='none',width=1.0)
for d in range(1,7): ax2.axvline(d*24+0.5,color='gray',lw=0.4,alpha=0.4)
for d,lbl in enumerate(DAY_LABELS): ax2.text(d*24+12,52,lbl,ha='center',fontsize=8.5,color='#555')
ax2.set_xlabel('Hour',fontsize=10);ax2.set_ylabel('Payoff ($/MWh per MW)',fontsize=11)
ax2.set_title('Spark Spread Option Payoff per Hour  (k=7)',fontsize=11,fontweight='bold')
ax2.set_xlim(0.5,T+0.5);ax2.set_ylim(0,56);ax2.grid(axis='y',alpha=0.25)
plt.tight_layout(pad=2.0)
plt.savefig('task4_spark_spread.png',dpi=150,bbox_inches='tight'); plt.close()

# FINAL

print("\n" + "=" * 60)
print("FINAL COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Model':<25} {'Gross Margin':>13} {'GM $/kW':>10} {'CF':>7} {'Starts':>8}")
print(f"  {'-'*65}")
print(f"  {'Task 2 CAISO MSG':<25} ${t2_gm:>12,.0f} ${t2_gm/cap[5]/1000:>9.4f} {t2_cf:>6.1f}% {t2_ns:>8}")
print(f"  {'Task 3 PJM Pseudo':<25} ${t3_gm_val:>12,.0f} ${t3_gm_val/cap[5]/1000:>9.4f} {s1['cf']:>6.1f}% {s1['ns']+s2['ns']:>8}")
print(f"  {'Spark spread k=7':<25} {'N/A (per MW)':>13} ${gm_per_kw_k7:>9.4f} {cf_k7:>6.1f}% {'N/A':>8}")
print(f"  {'k* for Task 2':<25} {'':>13} {'':>10} {'':>7} {k_star2:>7.3f}")
print(f"  {'k* for Task 3':<25} {'':>13} {'':>10} {'':>7} {k_star3:>7.3f}")
print("\nAll outputs saved: CCGT_CAISO.csv, CCGT_PSEUDO.csv, *.png")
