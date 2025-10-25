const API = "http://localhost:7860";

async function j(url){ const r=await fetch(url); if(!r.ok) throw new Error(r.statusText); return r.json(); }

function fmtMoney(x){ if(x===null||x===undefined||isNaN(x)) return "$—"; return "$"+(+x).toFixed(2); }
function fmtSigned(x, pct=false){
  if(x===null||x===undefined||isNaN(x)) return "—";
  const s = (x>=0?"+":"") + (pct? (x*100).toFixed(2)+"%" : (+x).toFixed(2));
  return s;
}
function classForSentiment(s){
  if(s==="Bullish") return "sentiment tag tag-bull";
  if(s==="Bearish") return "sentiment tag tag-bear";
  return "sentiment tag tag-neutral";
}
function pctFromDelta(delta, price){
  if(delta===null||delta===undefined) return null;
  if(price===null||price===undefined||price===0) return null;
  if(Math.abs(delta) < 1.0) return delta; // treat as return
  return delta / price;
}

let _ALL = []; // cache holdings for filtering

function renderSummary(sum){
  document.getElementById("sumTreasure").textContent = fmtMoney(sum.treasure || 0);
  const chg = sum.fortune_change || 0;
  const chgEl = document.getElementById("sumChange");
  chgEl.textContent = (chg>=0?"+":"") + fmtMoney(Math.abs(chg));
  chgEl.className = chg>=0 ? "text-2xl font-bold mt-1 glow-up" : "text-2xl font-bold mt-1 glow-down";
  const winds = sum.winds_pct || 0;
  const windsEl = document.getElementById("sumWinds");
  windsEl.textContent = (winds>=0?"+":"") + winds.toFixed(2) + "%";
  windsEl.className = winds>=0 ? "text-2xl font-bold mt-1 glow-up" : "text-2xl font-bold mt-1 glow-down";
  document.getElementById("sumCount").textContent = (sum.count||0) + " Holdings";
  document.getElementById("glance").textContent = `${sum.gainers||0} gainers • ${sum.losers||0} losers`;
}

function renderCards(holdings){
  const cont = document.getElementById("cards");
  cont.innerHTML = "";
  const tmpl = document.getElementById("cardTmpl");
  holdings.forEach(h => {
    const node = tmpl.content.cloneNode(true);
    node.querySelector(".ticker").textContent = h.ticker;
    node.querySelector(".company").textContent = h.company_name || "";
    const sEl = node.querySelector(".sentiment");
    sEl.textContent = h.sentiment || "Neutral";
    sEl.className = classForSentiment(h.sentiment);

    // Price & change
    node.querySelector(".price").textContent = fmtMoney(h.last_price || 0);
    const pct = pctFromDelta(h.delta1, h.last_price);
    const changeTxt = (h.change_val!=null? (h.change_val>=0?"+":"") + fmtMoney(Math.abs(h.change_val)) : "—")
                      + " (" + (pct!=null? ((pct>=0?"+":"")+(pct*100).toFixed(2)+"%") : "—") + ")";
    const chEl = node.querySelector(".change");
    chEl.textContent = changeTxt;
    chEl.className = "text-xs mt-1 " + ( (pct||0) >= 0 ? "glow-up" : "glow-down" );

    // Sparkline
    const ctx = node.querySelector(".spark").getContext("2d");
    const data = (h.forecast||[]).slice(0,7);
    new Chart(ctx, {
      type: "line",
      data: { labels: data.map((_,i)=>""+(i+1)), datasets: [{ data, tension: 0.35 }]},
      options: {
        plugins: { legend: { display:false } },
        scales: { x: { display:false }, y: { display:false } },
        elements: { point: { radius: 0 } }
      }
    });

    // Attach searchable text to dataset for filtering
    const root = node.firstElementChild;
    root.dataset.search = `${h.ticker} ${(h.company_name||"")}`.toLowerCase();
    cont.appendChild(node);
  });
}

async function load(){
  const dash = await j(`${API}/api/dashboard?steps=7`);
  _ALL = dash.holdings || [];
  renderSummary(dash.summary || {});
  renderCards(_ALL);
}

function filterCards(q){
  q = (q||"").toLowerCase().trim();
  const cards = document.querySelectorAll("#cards > .glass.card");
  if(!q){ cards.forEach(c => c.style.display=""); return; }
  cards.forEach(c => {
    const key = c.dataset.search || "";
    c.style.display = key.includes(q) ? "" : "none";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  load().catch(e => { console.error(e); alert("Backend not reachable. Start the API server first."); });
  const sb = document.getElementById("searchBox");
  sb.addEventListener("input", (e) => filterCards(e.target.value));
});
