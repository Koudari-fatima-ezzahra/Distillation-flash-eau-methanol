#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8000

# ---------------------------------------------------------------------------
# 1. DONNÉES THERMODYNAMIQUES   (A = méthanol, B = eau)
# ---------------------------------------------------------------------------
# Antoine : log10(Psat[mmHg]) = A - B / (C + T[°C])
ANTOINE = [
    (8.08097, 1582.271, 239.726),   # méthanol
    (8.07131, 1730.630, 233.426),   # eau
]
CP_L = [81.1, 75.4]        # J/mol/K   capacité calorifique liquide
H_VAP = [35.3, 40.7]       # kJ/mol    chaleur latente (approx. au point d'ébullition)
UNITS_TO_MMHG = {"atm": 760.0, "bar": 750.062, "kPa": 7.50062, "mmHg": 1.0}


def psat(i, T):
    A, B, C = ANTOINE[i]
    return 10 ** (A - B / (C + T))


def gammas(xA, model, L12, L21):
    """Coefficients d'activité de Wilson (1 = méthanol, 2 = eau)."""
    if model == "ideal":
        return 1.0, 1.0
    x1 = min(max(xA, 1e-12), 1.0)
    x2 = 1.0 - x1
    a = x1 + L12 * x2
    b = x2 + L21 * x1
    t = L12 / a - L21 / b
    return math.exp(-math.log(a) + x2 * t), math.exp(-math.log(b) - x1 * t)


def k_values(T, P_mmHg, xA, gm):
    g1, g2 = gammas(xA, gm["model"], gm["L12"], gm["L21"])
    return [g1 * psat(0, T) / P_mmHg, g2 * psat(1, T) / P_mmHg]


# ---------------------------------------------------------------------------
# 2. RACHFORD-RICE  (résolution par bissection)
# ---------------------------------------------------------------------------
def rr_function(V, z, K):
    return sum(z[i] * (K[i] - 1.0) / (1.0 + V * (K[i] - 1.0)) for i in range(2))


def solve_rachford_rice(z, K):
    """Retourne (V/F, état) avec état = liquide | biphasique | vapeur."""
    g0 = rr_function(0.0, z, K)
    g1 = rr_function(1.0, z, K)
    if g0 <= 0:
        return 0.0, "Liquide sous-refroidi (T < point de bulle)"
    if g1 >= 0:
        return 1.0, "Vapeur surchauffée (T > point de rosée)"
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rr_function(mid, z, K) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "Mélange liquide-vapeur (biphasique)"


def flash(zA, T, P_mmHg, gm):
    """Flash isotherme : boucle externe sur les coefficients d'activité."""
    z = [zA, 1.0 - zA]
    xA = zA
    V, etat = 0.0, ""
    for _ in range(500):
        K = k_values(T, P_mmHg, xA, gm)
        V, etat = solve_rachford_rice(z, K)
        x = [z[i] / (1.0 + V * (K[i] - 1.0)) for i in range(2)]
        sx = sum(x)
        x = [xi / sx for xi in x]
        if abs(x[0] - xA) < 1e-12:
            break
        xA = 0.3 * xA + 0.7 * x[0]
    y = [K[i] * x[i] for i in range(2)]
    sy = sum(y)
    y = [yi / sy for yi in y]
    return {"V": V, "x": x, "y": y, "K": K, "etat": etat,
            "gamma": list(gammas(x[0], gm["model"], gm["L12"], gm["L21"]))}


def bubble_T(zA, P, gm):
    """Température de bulle : sum(z_i K_i) = 1."""
    lo, hi = -50.0, 250.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        K = k_values(mid, P, zA, gm)
        if zA * K[0] + (1 - zA) * K[1] - 1.0 < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dew_T(zA, P, gm):
    """Température de rosée : sum(z_i / K_i) = 1 (itération sur la composition du liquide)."""
    xA = zA
    T = 80.0
    for _ in range(100):
        lo, hi = -50.0, 250.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            K = k_values(mid, P, xA, gm)
            if zA / K[0] + (1 - zA) / K[1] - 1.0 > 0:
                lo = mid
            else:
                hi = mid
        T = 0.5 * (lo + hi)
        K = k_values(T, P, xA, gm)
        x = [zA / K[0], (1 - zA) / K[1]]
        new = x[0] / sum(x)
        if abs(new - xA) < 1e-10:
            break
        xA = 0.5 * xA + 0.5 * new
    return T


# ---------------------------------------------------------------------------
# 3. SIMULATION COMPLÈTE
# ---------------------------------------------------------------------------
def simulate(d):
    zA = float(d["zA"])
    zB = float(d["zB"])
    if zA < 0 or zB < 0 or zA + zB <= 0:
        raise ValueError("Les fractions molaires doivent être positives.")
    s = zA + zB
    zA, zB = zA / s, zB / s                      # normalisation
    T0, Tf = float(d["T0"]), float(d["Tf"])
    F = float(d["F"])
    if F <= 0:
        raise ValueError("Le débit d'alimentation doit être > 0.")
    P_in = float(d["P"])
    if P_in <= 0:
        raise ValueError("La pression doit être > 0.")
    P = P_in * UNITS_TO_MMHG.get(d.get("Pu", "atm"), 760.0)
    M = [float(d["MA"]), float(d["MB"])]
    gm = {"model": d.get("model", "wilson"),
          "L12": float(d.get("L12", 0.5)), "L21": float(d.get("L21", 0.9))}

    # --- résultat au point de fonctionnement --------------------------------
    r = flash(zA, Tf, P, gm)
    V, L = r["V"] * F, (1 - r["V"]) * F
    x, y, K = r["x"], r["y"], r["K"]
    z = [zA, zB]

    # --- bilan enthalpique (H) -----------------------------------------------
    cp_mix = zA * CP_L[0] + zB * CP_L[1]
    q_sens = F * cp_mix * (Tf - T0) / 1000.0                      # kJ/h
    q_lat = V * (y[0] * H_VAP[0] + y[1] * H_VAP[1])               # kJ/h
    Q_kW = (q_sens + q_lat) / 3600.0

    # --- vérification MESH ------------------------------------------------------
    mesh = {
        "M": [F * z[i] - L * x[i] - V * y[i] for i in range(2)],
        "E": [y[i] - K[i] * x[i] for i in range(2)],
        "S": [sum(x) - 1.0, sum(y) - 1.0],
        "RR": rr_function(r["V"], z, K) if 0 < r["V"] < 1 else 0.0,
    }

    # --- courbe de chauffe T0 -> Tf (pour l'animation) ----------------------------
    n = 60
    curve = []
    for k in range(n + 1):
        T = T0 + (Tf - T0) * k / n
        c = flash(zA, T, P, gm)
        curve.append({"T": T, "V": c["V"], "x": c["x"][0], "y": c["y"][0]})

    # --- diagramme T-x-y à la pression P (80 points : courbes plus lisses) --------
    diag = {"x": [], "y": [], "T": []}
    N = 80
    for k in range(0, N + 1):
        xa = k / float(N)
        Tb = bubble_T(xa, P, gm)
        K = k_values(Tb, P, xa, gm)
        ya = K[0] * xa / (K[0] * xa + K[1] * (1 - xa))
        diag["x"].append(xa)
        diag["y"].append(ya)
        diag["T"].append(Tb)

    return {
        "ok": True,
        "z": z, "T0": T0, "Tf": Tf, "F": F, "P_mmHg": P,
        "etat": r["etat"],
        "VF": r["V"], "V": V, "L": L,
        "x": x, "y": y, "K": K, "gamma": r["gamma"],
        "psat": [psat(0, Tf), psat(1, Tf)],
        "Tb": bubble_T(zA, P, gm), "Td": dew_T(zA, P, gm),
        "mass": {
            "F": F * (zA * M[0] + zB * M[1]),
            "V": V * (y[0] * M[0] + y[1] * M[1]),
            "L": L * (x[0] * M[0] + x[1] * M[1]),
            "wV": y[0] * M[0] / (y[0] * M[0] + y[1] * M[1]),
            "wL": x[0] * M[0] / (x[0] * M[0] + x[1] * M[1]),
        },
        "Q_kW": Q_kW, "Q_kJh": q_sens + q_lat,
        "mesh": mesh, "curve": curve, "diag": diag,
    }


# ---------------------------------------------------------------------------
# 4. INTERFACE WEB (HTML + CSS + JavaScript, intégrés)
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flash eau–méthanol · Rachford-Rice & MESH</title>
<style>
:root{
  --paper:#eef1ee; --ink:#16263a; --muted:#5b6b7b; --line:#cfd8d4; --card:#fbfcfb;
  --meoh:#e07b12; --water:#1f78b4; --hot:#d6401f; --ok:#2f8f5b; --strip:#7a3fb0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.45}
header{padding:26px 32px 18px;border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 22px}
h1{margin:0;font-family:Georgia,"Times New Roman",serif;font-weight:700;font-size:28px;letter-spacing:-.3px}
header p{margin:0;color:var(--muted)}
.wrap{display:grid;grid-template-columns:330px 1fr;gap:22px;padding:22px 32px 40px;max-width:1800px;margin:auto}
@media(max-width:980px){.wrap{grid-template-columns:1fr;padding:16px}}
main{min-width:0}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start}
@media(max-width:1300px){.cols{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
.panel h2{margin:0 0 12px;font-size:16px;font-family:Georgia,serif}
.panel + .panel{margin-top:18px}
label{display:block;font-size:13px;color:var(--muted);margin:11px 0 4px}
.row{display:flex;gap:8px}
.row>*{flex:1;min-width:0}
input,select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:7px;
  background:#fff;font:inherit;color:var(--ink)}
input:focus,select:focus,button:focus-visible{outline:2px solid var(--water);outline-offset:1px}
.swA{border-left:5px solid var(--meoh)} .swB{border-left:5px solid var(--water)}
button{cursor:pointer;border:0;border-radius:8px;padding:12px 16px;font:inherit;font-weight:600}
.go{width:100%;margin-top:18px;background:var(--ink);color:#fff}
.go:hover{background:#0d1a29}
.ghost{background:#e3e9e6;color:var(--ink);padding:7px 12px;font-weight:500}
small{color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.kpi{border:1px solid var(--line);border-radius:9px;padding:12px 14px;background:#fff}
.kpi b{display:block;font-size:22px;font-family:Georgia,serif;margin-top:2px}
.kpi span{font-size:12.5px;color:var(--muted)}
.state{margin:0 0 14px;padding:9px 13px;border-radius:8px;background:#e5efe9;border:1px solid #bcd6c6;font-weight:600}
.state.warn{background:#fbeee6;border-color:#efc7ad}
canvas{width:100%;display:block;border-radius:8px;background:#fff;border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:13px}
.dotA,.dotB{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.dotA{background:var(--meoh)} .dotB{background:var(--water)}
.bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:10px;flex-wrap:wrap}
.bar input[type=range]{width:120px;padding:0}
.err{color:#b3261e;font-weight:600;margin-top:10px}
.eq{font-family:Georgia,serif;font-style:italic;background:#f1f4f2;border-radius:7px;padding:8px 11px;margin:10px 0 0;font-size:14px}
.eq b{font-style:normal}
.tEnr{color:var(--ok)} .tEpu{color:var(--strip)} .tQ{color:var(--hot)}
.stack > * + *{margin-top:16px}
</style>
</head>
<body>
<header>
  <h1>Distillation flash eau–méthanol</h1>
  
</header>

<div class="wrap">
  <!-- ======================= ENTRÉES ======================= -->
  <aside>
    <div class="panel">
      <h2>Alimentation</h2>
      <label>Fraction molaire du méthanol z<sub>A</sub></label>
      <input id="zA" class="swA" type="number" step="0.01" min="0" max="1" value="0.4">
      <label>Fraction molaire de l'eau z<sub>B</sub></label>
      <input id="zB" class="swB" type="number" step="0.01" min="0" max="1" value="0.6">
      <label>Débit d'alimentation F (mol/h)</label>
      <input id="F" type="number" step="1" min="0.001" value="50">
      <div class="row">
        <div><label>Masse molaire A (g/mol)</label><input id="MA" type="number" step="0.001" value="32.042"></div>
        <div><label>Masse molaire B (g/mol)</label><input id="MB" type="number" step="0.001" value="18.015"></div>
      </div>
    </div>
    <div class="panel">
      <h2>Conditions opératoires</h2>
      <div class="row">
        <div><label>T entrée (°C)</label><input id="T0" type="number" step="1" value="25"></div>
        <div><label>T chauffage (°C)</label><input id="Tf" type="number" step="1" value="80"></div>
      </div>
      <div class="row">
        <div><label>Pression</label><input id="P" type="number" step="0.01" min="0.001" value="1"></div>
        <div><label>Unité</label>
          <select id="Pu"><option>atm</option><option>bar</option><option>kPa</option><option>mmHg</option></select></div>
      </div>
      <label>Modèle thermodynamique</label>
      <select id="model">
        <option value="wilson">Wilson (mélange non idéal)</option>
        <option value="ideal">Loi de Raoult (idéal)</option>
      </select>
      <div class="row" id="wilsonBox">
        <div><label>Λ<sub>AB</sub></label><input id="L12" type="number" step="0.01" value="0.50"></div>
        <div><label>Λ<sub>BA</sub></label><input id="L21" type="number" step="0.01" value="0.90"></div>
      </div>
      <small>Les paramètres de Wilson sont des valeurs approchées, modifiables.</small>
      <button class="go" id="run">Lancer la simulation</button>
      <div class="err" id="err"></div>
    </div>
    <div class="panel">
      <h2>Droites opératoires (colonne)</h2>
      <label>Taux de reflux R = L/D</label>
      <input id="R" type="number" step="0.1" min="0.01" value="1.5">
      <div class="row">
        <div><label>x<sub>D</sub> (distillat)</label><input id="xD" type="number" step="0.01" min="0" max="1" value="0.95"></div>
        <div><label>x<sub>W</sub> (résidu)</label><input id="xW" type="number" step="0.01" min="0" max="1" value="0.05"></div>
      </div>
      <small>Mise à jour immédiate sur le diagramme x–y. Il faut x<sub>W</sub> &lt; z<sub>A</sub> &lt; x<sub>D</sub>.</small>
    </div>
  </aside>

  <!-- ======================= SORTIES ======================= -->
  <main>
    <div class="cols">
      <!-- colonne gauche : dessin 2D + résultats -->
      <div>
        <div class="panel">
          <div class="bar">
            <h2 style="margin:0">Simulation 2D du ballon flash</h2>
            <div>
              <small>Vitesse</small>
              <input id="speed" type="range" min="0.3" max="3" step="0.1" value="1">
              <button class="ghost" id="replay">Rejouer</button>
            </div>
          </div>
          <canvas id="sim" width="720" height="430"></canvas>
        </div>

        <div class="panel">
          <h2>Résultats au point de fonctionnement</h2>
          <div class="state" id="state">—</div>
          <div class="kpis" id="kpis"></div>
          <div class="stack" style="margin-top:18px">
            <div>
              <table id="tbl"></table>
              <div class="eq">Rachford-Rice : Σ z<sub>i</sub>(K<sub>i</sub>−1) / [1+(V/F)(K<sub>i</sub>−1)] = 0 &nbsp;avec&nbsp; K<sub>i</sub> = γ<sub>i</sub>P<sub>i</sub><sup>sat</sup>/P</div>
            </div>
            <div><table id="mesh"></table></div>
          </div>
        </div>
      </div>

      <!-- colonne droite : graphiques agrandis -->
      <div>
        <div class="panel">
          <h2>Diagramme T–x–y à la pression P</h2>
          <canvas id="txy" width="640" height="420"></canvas>
        </div>
        <div class="panel">
          <h2>Diagramme x–y : droite q, enrichissement, épuisement</h2>
          <canvas id="vt" width="640" height="520"></canvas>
          <div class="eq" id="colEq">—</div>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
const $ = id => document.getElementById(id);
const COL = {A:'#e07b12', B:'#1f78b4', ink:'#16263a', line:'#cfd8d4', hot:'#d6401f', enr:'#2f8f5b', epu:'#7a3fb0'};
let DATA = null, simT = 0, last = 0, running = false;
const f = (v,d=3) => Number(v).toFixed(d);

/* ---- liaison z_A + z_B = 1 ---- */
$('zA').addEventListener('input',()=>{ $('zB').value = f(Math.max(0,1-$('zA').value),4)*1; });
$('zB').addEventListener('input',()=>{ $('zA').value = f(Math.max(0,1-$('zB').value),4)*1; });
$('model').addEventListener('change',()=>{ $('wilsonBox').style.display = $('model').value==='wilson'?'flex':'none'; });
$('run').onclick = run;
$('replay').onclick = ()=>{ simT = 0; };

async function run(){
  $('err').textContent = '';
  const body = {};
  ['zA','zB','F','MA','MB','T0','Tf','P','Pu','model','L12','L21'].forEach(k=>body[k]=$(k).value);
  try{
    const r = await fetch('/api/flash',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    if(!d.ok){ $('err').textContent = d.error; return; }
    DATA = d; renderResults(d); resetSim(); if(!running){ running = true; requestAnimationFrame(loop); }
  }catch(e){ $('err').textContent = 'Erreur : ' + e.message; }
}

/* =========================== RÉSULTATS =========================== */
function renderResults(d){
  const st = $('state'); st.textContent = d.etat;
  st.className = 'state' + (d.VF>0 && d.VF<1 ? '' : ' warn');
  const k = [
    ['Vapeur V (mol/h)', f(d.V,2)], ['Liquide L (mol/h)', f(d.L,2)], ['Taux V/F', f(d.VF,4)],
    ['Point de bulle', f(d.Tb,1)+' °C'], ['Point de rosée', f(d.Td,1)+' °C'],
    ['Puissance Q', f(d.Q_kW,3)+' kW'],
    ['Débit massique V', f(d.mass.V,1)+' g/h'], ['Débit massique L', f(d.mass.L,1)+' g/h']
  ];
  $('kpis').innerHTML = k.map(a=>`<div class="kpi"><span>${a[0]}</span><b>${a[1]}</b></div>`).join('');
  const row=(n,i,c)=>`<tr><td><i class="${c}"></i>${n}</td><td>${f(d.z[i],4)}</td><td>${f(d.x[i],4)}</td><td>${f(d.y[i],4)}</td><td>${f(d.K[i],4)}</td><td>${f(d.gamma[i],3)}</td></tr>`;
  $('tbl').innerHTML = `<tr><th>Composant</th><th>z</th><th>x (liq.)</th><th>y (vap.)</th><th>K</th><th>γ</th></tr>`
    + row('<span class="dotA"></span>Méthanol (A)',0,'') + row('<span class="dotB"></span>Eau (B)',1,'');
  const m=d.mesh, e=v=>Number(v).toExponential(1);
  $('mesh').innerHTML = `<tr><th>Vérification MESH</th><th>A</th><th>B</th></tr>
    <tr><td>M : F·z − L·x − V·y</td><td>${e(m.M[0])}</td><td>${e(m.M[1])}</td></tr>
    <tr><td>E : y − K·x</td><td>${e(m.E[0])}</td><td>${e(m.E[1])}</td></tr>
    <tr><td>S : Σx − 1 , Σy − 1</td><td>${e(m.S[0])}</td><td>${e(m.S[1])}</td></tr>
    <tr><td>H : chaleur échangée</td><td colspan="2">${f(d.Q_kJh,1)} kJ/h</td></tr>
    <tr><td>Rachford-Rice : f(V/F)</td><td colspan="2">${e(m.RR)}</td></tr>`;
}

/* =========================== ANIMATION (dessin réduit 720 x 430) =========================== */
const cv = $('sim'), g = cv.getContext('2d'), W = cv.width, H = cv.height;
const DR = {x0:360, x1:540, top:55, bot:395};          // ballon
const YF = 240, YV = 95, YL = 365;                      // alimentation, sortie vapeur, sortie liquide
let liq=[], bub=[], vap=[], feed=[], out=[], cur={T:25,V:0,x:.4,y:.4,fl:0};
const RAMP = 9;                                         // durée du chauffage (s)

function resetSim(){
  simT = 0; bub=[]; vap=[]; feed=[]; out=[]; liq=[];
  for(let i=0;i<130;i++) liq.push({x:DR.x0+8+Math.random()*(DR.x1-DR.x0-16), y:DR.bot-8-Math.random()*230,
    vx:(Math.random()-.5)*30, vy:(Math.random()-.5)*30, u:Math.random()});
}
const ease = t => t<.5 ? 2*t*t : 1-Math.pow(-2*t+2,2)/2;

function interp(prog){
  const c = DATA.curve, p = prog*(c.length-1), i = Math.min(c.length-2, Math.floor(p)), a = p-i;
  const m = k => c[i][k]*(1-a)+c[i+1][k]*a;
  return {T:m('T'), V:m('V'), x:m('x'), y:m('y')};
}
const pick = (u,xA) => u<xA ? COL.A : COL.B;

function loop(ts){
  const dt = Math.min(.05, (ts-last)/1000 || .016); last = ts;
  if(DATA){
    simT += dt*Number($('speed').value);
    const prog = ease(Math.min(1, simT/RAMP));
    cur = interp(prog); cur.heat = prog;
    step(dt); draw(); drawTxy(); drawXY();
  }
  requestAnimationFrame(loop);
}

function level(){ return DR.bot - (DR.bot-DR.top)*(0.30 + 0.38*(1-cur.V)); }   // niveau du liquide

function step(dt){
  const lv = level(), speed = 25 + cur.T*1.6;
  // molécules du liquide
  liq.forEach(p=>{
    p.vx += (Math.random()-.5)*speed*dt*8; p.vy += (Math.random()-.5)*speed*dt*8;
    const s = Math.hypot(p.vx,p.vy), mx = speed; if(s>mx){ p.vx*=mx/s; p.vy*=mx/s; }
    p.x += p.vx*dt; p.y += p.vy*dt;
    if(p.x<DR.x0+6||p.x>DR.x1-6){ p.vx*=-1; p.x=Math.min(DR.x1-6,Math.max(DR.x0+6,p.x)); }
    if(p.y<lv+6||p.y>DR.bot-6){ p.vy*=-1; p.y=Math.min(DR.bot-6,Math.max(lv+6,p.y)); }
  });
  // alimentation (couleur = z)
  if(Math.random() < dt*7) feed.push({x:12,y:YF+(Math.random()-.5)*14,u:Math.random()});
  feed.forEach(p=>p.x += 105*dt); feed = feed.filter(p=>p.x<DR.x0+4);
  // bulles + vapeur
  if(cur.V>0.002 && Math.random() < dt*(10+80*cur.V)){
    bub.push({x:DR.x0+14+Math.random()*(DR.x1-DR.x0-28), y:DR.bot-10, r:2.5+Math.random()*4, ph:Math.random()*6, u:Math.random()});
  }
  bub.forEach(b=>{ b.y -= (55+b.r*8)*dt; b.ph += dt*5; b.x += Math.sin(b.ph)*14*dt; });
  bub = bub.filter(b=>{
    if(b.y <= level()){ vap.push({x:b.x, y:level()-4, c:pick(b.u,cur.y), s:0}); return false; } return true; });
  vap.forEach(p=>{
    if(p.s===0){ const dx = DR.x1-14-p.x, dy = YV-p.y, d = Math.hypot(dx,dy)||1;
      p.x += (dx/d*70+(Math.random()-.5)*50)*dt; p.y += (dy/d*70+(Math.random()-.5)*50)*dt;
      if(p.x>DR.x1-22 && p.y<YV+10) p.s=1; }
    else { p.x += 120*dt; p.y = YV+Math.sin(p.x*.08)*3; }
  });
  vap = vap.filter(p=>p.x<W-8);
  // sortie liquide
  if(cur.V<0.999 && Math.random() < dt*8*(1-cur.V)+dt*1.5) out.push({x:DR.x1-6,y:YL+(Math.random()-.5)*12,u:Math.random(),c:null});
  out.forEach(p=>{ p.x += 95*dt; if(!p.c) p.c = pick(p.u,cur.x); });
  out = out.filter(p=>p.x<W-8);
}

function pipe(x0,y0,x1,y1,w=18){
  g.strokeStyle='#d5dbd8'; g.lineWidth=w; g.lineCap='butt'; g.beginPath(); g.moveTo(x0,y0); g.lineTo(x1,y1); g.stroke();
  g.strokeStyle='#8fa0a6'; g.lineWidth=1.5;
  [-w/2,w/2].forEach(o=>{ g.beginPath();
    if(y0===y1){g.moveTo(x0,y0+o);g.lineTo(x1,y1+o);} else {g.moveTo(x0+o,y0);g.lineTo(x1+o,y1);} g.stroke(); });
}
function dot(x,y,r,c,a=1){ g.globalAlpha=a; g.fillStyle=c; g.beginPath(); g.arc(x,y,r,0,7); g.fill(); g.globalAlpha=1; }

function draw(){
  g.clearRect(0,0,W,H);
  // fond quadrillé léger (papier millimétré)
  g.strokeStyle='#eef2f0'; g.lineWidth=1;
  for(let x=0;x<W;x+=30){g.beginPath();g.moveTo(x,0);g.lineTo(x,H);g.stroke();}
  for(let y=0;y<H;y+=30){g.beginPath();g.moveTo(0,y);g.lineTo(W,y);g.stroke();}

  // tuyaux
  pipe(0,YF,DR.x0,YF); pipe(DR.x1,YV,W,YV); pipe(DR.x1,YL,W,YL);

  // chauffeur (échangeur) sur l'alimentation
  const hx=150,hw=150,hy=YF-48,hh=96, k=cur.heat;
  g.fillStyle='#f6f7f6'; g.strokeStyle=COL.ink; g.lineWidth=2;
  g.beginPath(); g.roundRect(hx,hy,hw,hh,10); g.fill(); g.stroke();
  const gl=g.createLinearGradient(hx,hy+hh,hx,hy+hh+60);
  gl.addColorStop(0,`rgba(214,64,31,${.12+.55*k})`); gl.addColorStop(1,'rgba(214,64,31,0)');
  g.fillStyle=gl; g.fillRect(hx,hy+hh,hw,60);
  g.strokeStyle=`rgb(${200+55*k},${130-90*k},${60-30*k})`; g.lineWidth=5; g.beginPath();   // serpentin
  for(let i=0;i<=hw-30;i+=2){ const yy=YF+Math.sin(i*.12)*22; i?g.lineTo(hx+15+i,yy):g.moveTo(hx+15+i,yy); } g.stroke();
  for(let i=0;i<7;i++){   // flammes
    const fx=hx+18+i*19, fh=(10+16*k)*(0.7+0.3*Math.sin(simT*9+i*2));
    g.fillStyle=`rgba(${240},${120+60*Math.sin(i+simT*6)*.5},30,${.35+.6*k})`;
    g.beginPath(); g.moveTo(fx-7,hy+hh+2); g.quadraticCurveTo(fx,hy+hh+2+fh*1.4,fx+7,hy+hh+2); g.fill();
  }
  g.fillStyle=COL.ink; g.font='600 13px Segoe UI,Arial'; g.textAlign='center';
  g.fillText('Chauffeur',hx+hw/2,hy-8);

  // ballon
  const lv = level();
  g.save(); g.beginPath(); g.roundRect(DR.x0,DR.top,DR.x1-DR.x0,DR.bot-DR.top,26); g.clip();
  g.fillStyle='#f7fafb'; g.fillRect(DR.x0,DR.top,DR.x1-DR.x0,DR.bot-DR.top);
  const lg=g.createLinearGradient(0,lv,0,DR.bot);
  lg.addColorStop(0,'rgba(120,180,220,.35)'); lg.addColorStop(1,'rgba(40,110,170,.55)');
  g.fillStyle=lg; g.beginPath(); g.moveTo(DR.x0,DR.bot);
  for(let x=DR.x0;x<=DR.x1;x+=4) g.lineTo(x, lv+Math.sin(x*.05+simT*3)*2.2);
  g.lineTo(DR.x1,DR.bot); g.closePath(); g.fill();
  liq.forEach(p=>dot(p.x,p.y,3.6,pick(p.u,cur.x),.9));
  bub.forEach(b=>{ g.strokeStyle='rgba(255,255,255,.95)'; g.lineWidth=1.6; g.beginPath(); g.arc(b.x,b.y,b.r,0,7); g.stroke();
     g.fillStyle='rgba(255,255,255,.25)'; g.fill(); });
  vap.forEach(p=>{ if(p.x<DR.x1) dot(p.x,p.y,3.2,p.c,.85); });
  g.restore();
  g.strokeStyle=COL.ink; g.lineWidth=3; g.beginPath(); g.roundRect(DR.x0,DR.top,DR.x1-DR.x0,DR.bot-DR.top,26); g.stroke();
  // particules dans les tuyaux
  vap.forEach(p=>{ if(p.x>=DR.x1) dot(p.x,p.y,3.2,p.c); });
  out.forEach(p=>dot(p.x,p.y,3.4,p.c));
  feed.forEach(p=>{ const hot = p.x>hx && p.x<hx+hw; dot(p.x,p.y,3.4,pick(p.u,DATA.z[0]));
    if(hot) dot(p.x,p.y,6,COL.hot,.14*k+.04); });

  // étiquettes
  g.fillStyle=COL.ink; g.font='600 14px Segoe UI,Arial'; g.textAlign='left';
  g.fillText(`F = ${f(DATA.F,1)} mol/h`,14,YF-24); g.font='13px Segoe UI,Arial';
  g.fillText(`z(méthanol) = ${f(DATA.z[0],2)}`,14,YF+34);
  const LX = W-172;
  g.font='600 14px Segoe UI,Arial'; g.fillText(`Vapeur  V = ${f(cur.V*DATA.F,1)} mol/h`,LX,YV-24);
  g.font='13px Segoe UI,Arial'; g.fillText(`y(méthanol) = ${f(cur.y,3)}`,LX,YV+34);
  g.font='600 14px Segoe UI,Arial'; g.fillText(`Liquide  L = ${f((1-cur.V)*DATA.F,1)} mol/h`,LX,YL-24);
  g.font='13px Segoe UI,Arial'; g.fillText(`x(méthanol) = ${f(cur.x,3)}`,LX,YL+34);

  // thermomètre
  const tx=45, ty0=330, ty1=410, frac=(cur.T-DATA.T0)/((DATA.Tf-DATA.T0)||1);
  g.fillStyle='#fff'; g.strokeStyle=COL.ink; g.lineWidth=2; g.beginPath(); g.roundRect(tx-7,ty0-80,14,ty1-ty0+80,7); g.fill(); g.stroke();
  g.fillStyle=COL.hot; g.fillRect(tx-4,ty1-(ty1-ty0+60)*Math.max(.03,Math.min(1,frac)),8,(ty1-ty0+60)*Math.max(.03,Math.min(1,frac)));
  dot(tx,ty1+4,10,COL.hot); g.strokeStyle=COL.ink; g.beginPath(); g.arc(tx,ty1+4,10,0,7); g.stroke();
  g.fillStyle=COL.ink; g.font='700 22px Georgia,serif'; g.textAlign='left'; g.fillText(f(cur.T,1)+' °C',tx+22,ty0-18);
  g.font='13px Segoe UI,Arial'; g.fillText(cur.V<=0.0005?'liquide seul':(cur.V>=.9995?'vapeur seule':'ébullition partielle'),tx+22,ty0+2);
  // légende
  dot(DR.x0+16,DR.top-20,5,COL.A); g.font='13px Segoe UI,Arial'; g.fillStyle=COL.ink; g.fillText('méthanol',DR.x0+26,DR.top-15);
  dot(DR.x0+100,DR.top-20,5,COL.B); g.fillText('eau',DR.x0+110,DR.top-15);
}

/* =========================== GRAPHIQUES =========================== */
function axes(c, xr, yr, xl, yl){
  const x=c.getContext('2d'), w=c.width, h=c.height, m={l:56,r:16,t:16,b:46};
  x.clearRect(0,0,w,h); x.font='13px Segoe UI,Arial'; x.fillStyle=COL.ink; x.strokeStyle=COL.line; x.lineWidth=1;
  const X=v=>m.l+(v-xr[0])/(xr[1]-xr[0])*(w-m.l-m.r), Y=v=>h-m.b-(v-yr[0])/(yr[1]-yr[0])*(h-m.t-m.b);
  for(let i=0;i<=5;i++){ const xv=xr[0]+(xr[1]-xr[0])*i/5, yv=yr[0]+(yr[1]-yr[0])*i/5;
    x.beginPath(); x.moveTo(X(xv),m.t); x.lineTo(X(xv),h-m.b); x.moveTo(m.l,Y(yv)); x.lineTo(w-m.r,Y(yv)); x.stroke();
    x.textAlign='center'; x.fillText(xv.toFixed(xr[1]<=1?1:0),X(xv),h-m.b+17);
    x.textAlign='right'; x.fillText(yv.toFixed(yr[1]<=1.01?1:0),m.l-7,Y(yv)+4); }
  x.textAlign='center'; x.fillText(xl,(w+m.l)/2,h-8);
  x.save(); x.translate(14,h/2); x.rotate(-Math.PI/2); x.fillText(yl,0,0); x.restore();
  return {x,X,Y};
}
function line(a,xs,ys,col,wd=2.4,dash=[]){ a.x.strokeStyle=col; a.x.lineWidth=wd; a.x.setLineDash(dash); a.x.beginPath();
  xs.forEach((v,i)=>i?a.x.lineTo(a.X(v),a.Y(ys[i])):a.x.moveTo(a.X(v),a.Y(ys[i]))); a.x.stroke(); a.x.setLineDash([]); }
function pt(a,x,y,col,r=5){ a.x.fillStyle=col; a.x.beginPath(); a.x.arc(a.X(x),a.Y(y),r,0,7); a.x.fill(); a.x.strokeStyle='#fff'; a.x.lineWidth=1.5; a.x.stroke(); }

function drawTxy(){
  const d=DATA.diag, Ts=d.T, a=axes($('txy'),[0,1],[Math.floor(Math.min(...Ts)/5)*5-5,Math.ceil(Math.max(...Ts)/5)*5+5],'fraction molaire de méthanol','T (°C)');
  line(a,d.x,Ts,COL.B,2.8); line(a,d.y,Ts,COL.A,2.8);
  a.x.font='600 13px Segoe UI,Arial';
  a.x.fillStyle=COL.B; a.x.textAlign='left'; a.x.fillText('liquide (x)',a.X(.04),a.Y(Ts[0])+18);
  a.x.fillStyle=COL.A; a.x.fillText('vapeur (y)',a.X(.62),a.Y(Ts[Math.floor(Ts.length*.6)])-28);
  line(a,[DATA.z[0],DATA.z[0]],[DATA.T0,DATA.Tf],'#8a98a3',1.4,[5,4]);
  if(cur.V>0.0005 && cur.V<0.9995){ line(a,[cur.x,cur.y],[cur.T,cur.T],COL.hot,2.4); pt(a,cur.x,cur.T,COL.B,6); pt(a,cur.y,cur.T,COL.A,6); }
  pt(a,DATA.z[0],cur.T,COL.ink,5);
}

/* ---- inversion de la courbe d'équilibre : x = f(y) ---- */
function eqX(y){
  const d=DATA.diag; if(y<=0) return 0; if(y>=1) return 1;
  for(let i=1;i<d.y.length;i++){
    if(d.y[i]>=y){ const t=(y-d.y[i-1])/((d.y[i]-d.y[i-1])||1); return d.x[i-1]+t*(d.x[i]-d.x[i-1]); }
  }
  return 1;
}

/* ---- droites d'enrichissement / d'épuisement + droite q (flash) ----
   Enrichissement : y = R/(R+1)·x + xD/(R+1)       (passe par (xD,xD))
   Droite q (flash): y = −(L/V)·x + F·z/V          (passe par (z,z), pente −L/V)
   Épuisement     : droite joignant (xW,xW) à l'intersection I des deux autres */
function column(){
  const R=Number($('R').value), xD=Number($('xD').value), xW=Number($('xW').value), z=DATA.z[0], vf=cur.V;
  if(!(R>0)||!(xD>z)||!(xW<z)||xD>1||xW<0) return {err:'Il faut  x_W < z < x_D  et R > 0.'};
  const a=R/(R+1), b=xD/(R+1);
  let xi, yi;
  if(vf<=0.0005){ xi=z; yi=a*z+b; }                         // droite q verticale (liquide)
  else if(vf>=0.9995){ yi=z; xi=(z-b)/a; }                  // droite q horizontale (vapeur)
  else { xi=(z/vf-b)/(a+(1-vf)/vf); yi=a*xi+b; }
  if(!(xi>xW && xi<xD)) return {err:'Intersection hors du domaine [x_W ; x_D] : modifier R, x_D ou x_W.'};
  const m=(yi-xW)/(xi-xW), c=xW-m*xW;                       // épuisement : y = m·x + c
  let Rmin=null;
  if(vf>0.0005 && vf<0.9995 && cur.y>cur.x) Rmin=(xD-cur.y)/(cur.y-cur.x);
  // étages théoriques (escalier de McCabe-Thiele)
  const st=[[xD,xD]]; let x=xD, y=xD, n=0, pinch=false;
  while(n<60){
    const xn=eqX(y); st.push([xn,y]); n++;
    if(x-xn<1e-6){ pinch=true; break; }
    if(xn<=xW) break;
    const yn = xn>xi ? a*xn+b : m*xn+c;
    st.push([xn,yn]); x=xn; y=yn;
  }
  return {R,xD,xW,a,b,m,c,xi,yi,Rmin,st,n,pinch};
}

let lastEq='';
function drawXY(){
  const d=DATA.diag, a=axes($('vt'),[0,1],[0,1],'x méthanol (liquide)','y méthanol (vapeur)');
  const zA=DATA.z[0], biph = cur.V>0.0005 && cur.V<0.9995;
  line(a,[0,1],[0,1],'#8a98a3',1.4,[5,4]);                       // diagonale y = x
  const C=column();
  if(!C.err){ line(a,C.st.map(p=>p[0]),C.st.map(p=>p[1]),'#8c99a3',1.2); }      // escalier
  line(a,d.x,d.y,COL.A,3);                                       // courbe d'équilibre
  if(!C.err){
    line(a,[C.xD,C.xi],[C.xD,C.yi],COL.enr,2.8);                 // enrichissement
    line(a,[C.xW,C.xi],[C.xW,C.yi],COL.epu,2.8);                 // épuisement
    line(a,[zA,C.xi],[zA,C.yi],COL.hot,2.2,[6,4]);               // droite q (flash)
    pt(a,C.xD,C.xD,COL.enr,5); pt(a,C.xW,C.xW,COL.epu,5); pt(a,C.xi,C.yi,COL.ink,5);
  }
  if(biph){
    if(C.err) line(a,[zA,cur.x],[zA,cur.y],COL.hot,2,[6,4]);
    pt(a,cur.x,cur.y,COL.hot,6);                                  // point d'équilibre du flash (x ; y)
  }
  pt(a,zA,zA,COL.ink,5);                                          // alimentation (z, z)
  const t=a.x; t.textAlign='left'; t.font='13px Segoe UI,Arial';
  t.fillStyle=COL.ink; t.fillText('alimentation z',a.X(zA)+8,a.Y(zA)+17);
  t.fillStyle=COL.A; t.fillText('équilibre',a.X(.5),a.Y(.93));
  // légende (bas droite)
  const lg=(txt,col,yv)=>{ t.fillStyle=col; t.fillRect(a.X(.50),a.Y(yv)-4,18,4); t.fillText(txt,a.X(.50)+24,a.Y(yv)+1); };
  lg('enrichissement',COL.enr,.34); lg('épuisement',COL.epu,.28); lg('droite q (flash)',COL.hot,.22);
  if(!C.err){ t.fillStyle='#6b7a85'; t.fillText('étages théoriques : '+C.n+(C.pinch?' (pincement)':''),a.X(.50),a.Y(.15)); }
  if(biph){ t.fillStyle=COL.hot; t.fillText('(x ; y) = ('+f(cur.x,3)+' ; '+f(cur.y,3)+')',a.X(.50),a.Y(.08)); }

  // équations
  let html;
  if(C.err){ html='<span class="tQ">'+C.err+'</span>'; }
  else {
    const q = biph||cur.V<=0.0005||cur.V>=0.9995 ? (cur.V<=0.0005?'q = 1 (liquide) : x = z' : (cur.V>=0.9995?'q = 0 (vapeur) : y = z' : 'y = −'+f((1-cur.V)/cur.V,3)+'·x + '+f(zA/cur.V,3))) : '';
    html='<span class="tEnr"><b>Enrichissement</b> : y = '+f(C.a,3)+'·x + '+f(C.b,3)+'</span><br>'
      +'<span class="tEpu"><b>Épuisement</b> : y = '+f(C.m,3)+'·x '+(C.c<0?'− ':'+ ')+f(Math.abs(C.c),3)+'</span><br>'
      +'<span class="tQ"><b>Droite q (flash)</b> : '+q+'</span><br>'
      +'Intersection : ('+f(C.xi,3)+' ; '+f(C.yi,3)+')'+(C.Rmin!=null?' · R<sub>min</sub> ≈ '+f(C.Rmin,2):'');
  }
  if(html!==lastEq){ $('colEq').innerHTML=html; lastEq=html; }
}

$('wilsonBox').style.display='flex';
run();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 5. SERVEUR WEB
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, payload):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))

    def do_POST(self):
        if self.path != "/api/flash":
            self._send(404, "application/json", b'{"ok": false}')
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            out = simulate(data)
        except Exception as e:  # erreur de saisie ou de calcul
            out = {"ok": False, "error": "Entrée invalide : %s" % e}
        self._send(200, "application/json", json.dumps(out).encode("utf-8"))

    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:%d" % PORT
    print("Application disponible sur", url, "  (Ctrl+C pour arrêter)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")


if __name__ == "__main__":
    main()